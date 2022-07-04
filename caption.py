#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os

import torch
import numpy as np
from fairseq import utils,tasks
from fairseq import checkpoint_utils
from utils.eval_utils import eval_step
from tasks.mm_tasks.caption import CaptionTask
from models.ofa import OFAModel
from PIL import Image
import argparse
import cv2

def encode_text(text, length=None, append_bos=False, append_eos=False):
    s = task.tgt_dict.encode_line(
        line=task.bpe.encode(text),
        add_if_not_exist=False,
        append_eos=False
    ).long()
    if length is not None:
        s = s[:length]
    if append_bos:
        s = torch.cat([bos_item, s])
    if append_eos:
        s = torch.cat([s, eos_item])
    return s

# Construct input for caption task
def construct_sample(image: Image):
    patch_image = patch_resize_transform(image).unsqueeze(0)
    patch_mask = torch.tensor([True])
    src_text = encode_text(" what does the image describe?", append_bos=True, append_eos=True).unsqueeze(0)
    src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])
    sample = {
        "id":np.array(['42']),
        "net_input": {
            "src_tokens": src_text,
            "src_lengths": src_length,
            "patch_images": patch_image,
            "patch_masks": patch_mask
        }
    }
    return sample

# Function to turn FP32 to FP16
def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="image captioning")

    parser.add_argument("--data", "-d", help="Path to image folder")

    args = parser.parse_args()

    DATA_PATH = args.data


    # Register caption task
    tasks.register_task('caption', CaptionTask)

    # turn on cuda if GPU is available
    use_cuda = torch.cuda.is_available()
    # use fp16 only when GPU is available
    use_fp16 = False


    # In[3]:


    # Load pretrained ckpt & config
    overrides={"bpe_dir":"utils/BPE", "eval_cider":False, "beam":5, "max_len_b":16, "no_repeat_ngram_size":3, "seed":7}
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths('checkpoints/caption_base_best.pt'),
            arg_overrides=overrides
        )

    # Move models to GPU
    for model in models:
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Initialize generator
    generator = task.build_generator(models, cfg.generation)


    # In[4]:


    # Image transform
    from torchvision import transforms
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((cfg.task.patch_image_size, cfg.task.patch_image_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Text preprocess
    bos_item = torch.LongTensor([task.src_dict.bos()])
    eos_item = torch.LongTensor([task.src_dict.eos()])
    pad_idx = task.src_dict.pad()

    image_paths = []

    # In[39]:

    save_folder = "captioned/"
    isExist = os.path.exists(save_folder)
    # print(isExist)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(save_folder)
        # print("The new directory is created!")

    for dirname, _, filenames in os.walk(DATA_PATH):
        for filename in filenames:
            raw_path = os.path.join(dirname, filename)
            image = Image.open(raw_path)
            raw_filename = filename.split(".")[0]
            save_filename = raw_filename + "_cap.jpg"

            # Construct input sample & preprocess for GPU if cuda available
            sample = construct_sample(image)
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample

            with torch.no_grad():
                result, scores = eval_step(task, generator, models, sample)

            display_image = cv2.imread(raw_path)
            cv2.putText(display_image, result[0]['caption'], (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 100, 0), 2)

            filename = save_folder + raw_filename + "_cap.jpg"

            cv2.imwrite(filename, display_image)

# display(image)
# print('Caption: {}'.format(result[0]['caption']))

