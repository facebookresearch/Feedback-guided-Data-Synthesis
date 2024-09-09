# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import argparse
import pandas as pd
import submitit
from utils_gen import get_classifier, get_pipe
from utils_gen import generate_single_sample, mkdir_if_needed
from utils_gen import remove_if_corrupted, save_image
from submitit.helpers import Checkpointable


def generate(args):
    csv_path = os.path.join(args.output_dir, f'fakes.csv')
    df = pd.read_csv(csv_path)
    # shuffle such that all gpus dont start from the same sample.
    df = df.sample(frac=1)
    print('preping the pipe and classifier...')
    fg_classifier, fg_preprocessing = get_classifier(args, df)
    pipe = get_pipe()

    n_total_read = 0
    n_saved = 0
    n_corrupted = 0
    n_skipped = 1
    for index, row in df.iterrows():
        n_total_read += 1
        if index % args.num_gpus == args.gpu_id:
            mkdir_if_needed(args, row)
            if not os.path.exists(row['image_save_path']):
                cls_directory = row['cls_directory']
                current_files = [os.path.join(cls_directory, f) for f in os.listdir(cls_directory) 
                                 if (os.path.isfile(os.path.join(cls_directory, f)) and '.png' in f)]
                if int(row['count_per_cls']) > len(current_files):
                    image, criteria_final_value = generate_single_sample(
                        args, row,
                        pipe, fg_classifier, fg_preprocessing)
                    if not np.isnan(criteria_final_value):
                        success = save_image(image, row, criteria_final_value)
                        n_saved += success
                        n_corrupted += (1 - success)
                    else:
                        print('Skipping saving! Generated image is black!')
                        print('Trying one more time')
                        image, criteria_final_value = generate_single_sample(
                            args, row,
                            pipe, fg_classifier, fg_preprocessing)
                        if not np.isnan(criteria_final_value):
                            success = save_image(image, row, criteria_final_value)
                            n_saved += success
                            n_corrupted += (1 - success)
                        else:
                            n_corrupted += 1
                else:
                    print('Skipping, already enough sampels for this class!')
                    n_skipped += 1
            else:
                print('Image already exists!')
                n_skipped += 1
            remove_if_corrupted(row['image_save_path'], row["img_id"])    


class Runner(Checkpointable):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        try:
            job_env = submitit.JobEnvironment()
            job_id = int(job_env.array_job_id)
            num_gpus = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))
            self.gpu_id = int(job_env.array_task_id)
        except Exception:
            job_id = 0
            num_gpus = 1
            self.gpu_id = 0
        print(f"job id: {job_id}, Number of gpus: {num_gpus}, gpu id: {self.gpu_id}")
        self.args.gpu_id = self.gpu_id
        self.args.num_gpus = num_gpus
        generate(self.args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit_it', action='store_true', default=False, help='submit to cluster')
    parser.add_argument('--submitit_path', type=str)
    parser.add_argument('--slurm_partition', type=str)
    parser.add_argument('--output_dir', type=str,
                        help='contains csv files + classifier checkpoints + fake images')
    parser.add_argument('--timeout_min', type=int, default=100)
    parser.add_argument('--num_gpus', type=int, default=2)

    args = parser.parse_args()
    list_args = [args] * args.num_gpus
    if not args.submit_it:
        for args_ in list_args:
            Runner(args_)()
    else:
        executor = submitit.AutoExecutor(folder=args.submitit_path, slurm_max_num_timeout=30)
        executor.update_parameters(
            gpus_per_node=1, array_parallelism=512,
            tasks_per_node=1, cpus_per_task=1, nodes=1,
            timeout_min=args.timeout_min,
            slurm_partition=args.slurm_partition, slurm_signal_delay_s=120)
        with executor.batch():
            for args_ in list_args:
                runner = Runner(args_)
                job = executor.submit(runner)
