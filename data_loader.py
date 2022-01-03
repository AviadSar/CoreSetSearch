import os
import torch
from datasets import load_dataset
from core_set_finder import find_core_set
from arg_parser import parse_data_args


def filter_snli(dataset, cache_dir):
    cache_dir = os.path.join(cache_dir, 'all_annotated')
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)

    cache_file_names = {'train': os.path.join(cache_dir, 'train'),
                        'validation': os.path.join(cache_dir, 'valid'),
                        'test': os.path.join(cache_dir, 'test')}
    dataset = dataset.filter(lambda label: False if label == -1 else True,
                             input_columns='label',
                             cache_file_names=cache_file_names)
    return dataset


def load(args):
    dataset_name = args.dataset_name
    cache_dir = args.cache_path + dataset_name

    dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    dataset = dataset.shuffle(seed=42)

    if dataset_name == 'snli':
        dataset = filter_snli(dataset, cache_dir)
        dataset = find_core_set(dataset, args)

    return dataset


if __name__ == '__main__':
    args = parse_data_args()
    load(args)
