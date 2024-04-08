# This source code is adapted from:
# MCC: https://github.com/facebookresearch/MCC
import argparse
import glob
import os
import sys
sys.path.append('../')
from collections import defaultdict

import torch

from util.dataset_utils import get_dataset_map


def main(args):
    dataset_cache_folder = args.dataset_cache #'../dataset_cache'
    if not os.path.isdir(dataset_cache_folder):
        os.mkdir(dataset_cache_folder)
    all_categories = [c.split('/')[-1] for c in list(glob.glob(args.dataset_path + '/*')) if not c.endswith('.json')]

    for category in all_categories:
        print(f'Loading dataset map ({category})')
        dataset_map = get_dataset_map(
            args.dataset_path,
            category,
            args.subset,
        )

        for split in ['train', 'val', 'test']:
            dataset = dataset_map[split]
            seq_name2idx = defaultdict(list)
            for i, ann in enumerate(dataset.frame_annots):
                #print(i, ann)
                seq_name2idx[ann["frame_annotation"].sequence_name].append(i)
            dataset.seq_name2idx = seq_name2idx
            torch.save(dataset, f'{dataset_cache_folder}/{category}_{split}.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MCC-HO', add_help=False)
    parser.add_argument('--dataset_path', required=True, type=str, help='path to dataset')
    parser.add_argument('--dataset_cache', default='../dataset_cache', required=True, type=str, help='path to dataset cache (for training)')
    parser.add_argument('--subset', default='test', required=True, type=str, help='dataset subset name')
    args = parser.parse_args()
    main(args)
