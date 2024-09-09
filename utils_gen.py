# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import torch
import pandas as pd
from PIL import Image
from diffusers import DiffusionPipeline
from transformers import CLIPFeatureExtractor, CLIPModel
import torchvision

torch.backends.cuda.matmul.allow_tf32 = True

def get_pipe():
    torch_dtype = torch.float16
    feature_extractor = CLIPFeatureExtractor.from_pretrained(
        "CLIP_MODEL_PATH")
    clip_model = CLIPModel.from_pretrained(
        "CLIP_MODEL_PATH",
        torch_dtype=torch_dtype)

    pipe = DiffusionPipeline.from_pretrained(
                "LDM_MODEL_PATH",
                custom_pipeline="fg_pipe.py",
                clip_model=clip_model,
                feature_extractor=feature_extractor,
                torch_dtype=torch_dtype,
                local_files_only=True)
    pipe = pipe.to("cuda")
    return pipe


def get_classifier(args, df):
    torch_dtype = torch.float16
    weights = torchvision.models.ResNet50_Weights.DEFAULT
    fg_classifier = torchvision.models.resnet50(weights=weights)
    fg_classifier.eval()
    fg_classifier.cuda()
    fg_classifier.to(torch_dtype)
    fg_preprocessing = torchvision.transforms.Compose(
        [torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    return fg_classifier, fg_preprocessing


def mkdir_if_needed(args, row):
    folder_path = os.path.join(
        args.output_dir,
        row['class_folder_name'])
    image_save_path = row['image_save_path']
    assert folder_path == os.path.dirname(image_save_path)
    try:
        os.makedirs(folder_path)
    except FileExistsError:
        # directory already exists
        pass

def remove_if_corrupted(file, img_id):
    try:
        if os.path.exists(file):
            image = Image.open(file)
            image.close()
            print('And image file is just fine!')
    except Image.UnidentifiedImageError:
        print('But image file is broken!')
        try:
            os.remove(file)
            os.remove(file.split('.png')[0] + '.txt')
            print('removed corrupted image!')
        except FileNotFoundError:
            print('could not remove the corrupted image!')
    except OSError:
        print('file already deleted by another gpu')

def save_image(image, row, criteria_final_value):
    try:
        image = image.resize((256, 256))
        image.save(row['image_save_path'])
        print(row['image_save_path'])
        with open(row['image_save_path'].split('.png')[0] + '.txt', 'w') as file:
            file.write(str(criteria_final_value))
        print('Successfully saved!')
        return 1
    except Exception as e:
        print(e)
        return 0

def generate_single_sample(args, row, pipe, fg_classifier, fg_preprocessing):
    generator = torch.Generator(device='cuda')
    # generator.manual_seed(seed)
    generator.seed()
    out, criteria_final_value = pipe(
        prompt=row['prompt'],
        cfg=row['cfg'],
        fg_criterion=row['fg_criterion'],
        fg_scale=row['fg_scale'],
        fg_classifier=fg_classifier,
        fg_preprocessing=fg_preprocessing,
        generator=generator,
        num_inference_steps=30,
        num_images_per_prompt=1,
        cls_index=row['cls_index'],
        guidance_freq=5)
    image = out[0]
    return image, criteria_final_value
