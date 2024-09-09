# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import json
import pandas as pd


def pre_populate_csv(fg_criterion, fg_scale, cfg_scale, output_dir, prompt_type, max_count):
   info_dict = json.load(open("imagenet_info.json"))
   csv_file_path = os.path.join(output_dir, 'fakes' + '.csv')
   data = []
   img_id = 0
   fakes_paths = {}
   fg_scale_ = fg_scale

   for cls_ in info_dict.keys():
      count_per_cls = max(max_count - info_dict[cls_]['count'], 1)
      for _ in range(count_per_cls):
         cls_directory = os.path.join(
            output_dir, info_dict[cls_]['cls_folder'])
         image_save_path = os.path.join(cls_directory, f'{img_id}.png')
          
         row = {
            'prompt': info_dict[cls_][prompt_type],
            'cls_name': info_dict[cls_]['text'],
            'full_text': info_dict[cls_]['full_txt'],
            'class_folder_name': info_dict[cls_]['cls_folder'],
            'img_id': img_id,
            'cfg': cfg_scale,
            'fg_criterion': fg_criterion,
            'fg_scale': fg_scale_,
            'cls_index': int(cls_),
            'image_save_path': image_save_path,
            'count_per_cls': count_per_cls,
            'cls_directory': cls_directory
         }
         class_folder_name = info_dict[cls_]['cls_folder']
         if class_folder_name in fakes_paths:
            fakes_paths[class_folder_name] += [image_save_path]
         else:
            fakes_paths[class_folder_name] = []
         data.append(row)
         img_id += 1
      try:
         os.makedirs(os.path.dirname(image_save_path))
      except FileExistsError:
         # directory already exists
         pass

   df = pd.DataFrame(data)
   df = df.sample(frac=1)
   df.to_csv(csv_file_path, index=False)
   print(f'csv saved at {csv_file_path}..')

   return fakes_paths


if __name__ == "__main__":
    pre_populate_csv(fg_criterion='entropy',
                     fg_scale=0.03,
                     cfg_scale=5,
                     output_dir='tmp',
                     prompt_type='text',
                     max_count=1300)
