import argparse
import json
import os
import torch
from trainer_args import TrainerArgs
from data_args import DataArgs


def update_training_args_from_work_env(args):
    if args.data_args.dataset_name == 'snli':
        args.num_labels = 3
    if args.data_args.work_env == 'home':
        if args.model_name == 'roberta-base':
            args.batch_size = 4
    elif args.data_args.work_env == 'hadassah':
        if args.model_name == 'roberta-base':
            args.batch_size = 16
    else:
        raise ValueError('Invalid value for --work_env: {}'.format(args.work_env))


def update_data_args_from_work_env(args):
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    if args.work_env == 'home':
        args.cache_path = 'C:\\my_documents\\datasets\\'
        if args.core_set_model_name == 'roberta-base':
            args.core_set_batch_size = 4
    elif args.work_env == 'hadassah':
        args.cache_path = '/home/aviad/Documents'
        if args.core_set_model_name == 'roberta-base':
            args.core_set_batch_size = 16
    else:
        raise ValueError('Invalid value for --work_env: {}'.format(args.work_env))


def parse_trainer_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--json_file",
        help="path to a json file to load arguments from",
        type=str,
        default=None
    )
    parser.add_argument(
        "--model_name",
        help="name of the model to load from the web",
        type=str,
        default="roberta-base"
    )
    parser.add_argument(
        '--batch_size',
        help='number of samples in each batch of training/evaluating',
        type=int,
        default=None,
    )
    parser.add_argument(
        '--split_ratio',
        help='percentage of samples in each batch of training/evaluating',
        type=int,
        nargs=3,
        default=[0.0005, 0.01, 0.01],
    )
    parser.add_argument(
        '--logging_steps',
        help='training steps between logs',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--num_epochs',
        help='training steps between evaluations',
        type=int,
        default=20,
    )
    parser.add_argument(
        '--eval',
        help='if true, only evaluate on dev set',
        type=bool,
        default=False,
    )
    parser.add_argument(
        '--test',
        help='if true, only evaluate test set',
        type=bool,
        default=False,
    )

    args = parser.parse_args()
    if args.json_file:
        with open(args.json_file, 'r') as json_file:
            json_data = json.load(json_file)
            args = TrainerArgs(json_data)
    args.data_args = parse_data_args()
    update_training_args_from_work_env(args)
    update_data_args_from_work_env(args.data_args)
    return args


def parse_data_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--json_file",
        help="path to a json file to load arguments from",
        type=str,
        default=None
    )
    parser.add_argument(
        "--dataset_name",
        help="dataset name",
        type=str,
        default="snli"
    )
    parser.add_argument(
        "--work_env",
        help="where i work from: home, schwartz-lab, haddasah, colab etc.",
        type=str,
        default="home"
    )
    parser.add_argument(
        "--core_set_method",
        help="the method to create a core-set with",
        type=str,
        default=None
    )
    parser.add_argument(
        "--core_set_size",
        help="desired size of the core-set, as a fraction of the whole set",
        type=float,
        default=1
    )
    parser.add_argument(
        "--core_set_model_name",
        help="name of the model to to use for core set searching",
        type=str,
        default="roberta-base"
    )
    parser.add_argument(
        "--core_set_batch_size",
        help="batch size to use while processing the dataset to create a core set",
        type=int,
        default=None
    )
    parser.add_argument(
        "--device",
        help="the device to use (cuda or cpu)",
        type=str,
        default=None
    )

    args = parser.parse_args()
    if args.json_file:
        with open(args.json_file, 'r') as json_file:
            json_data = json.load(json_file)
            args = DataArgs(json_data)
    update_data_args_from_work_env(args)
    return args