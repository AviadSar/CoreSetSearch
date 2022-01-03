import torch
import transformers
import datasets
import time
import dataset_classes
import os
from model_loader import get_model_and_tokenizer_from_args
from data_tokenizer import tokenize
from greedy_k_means import GreedyKMeans


CORE_SET_BASE_DIR = 'core_sets'


def get_core_set_dir_from_args(args):
        core_set_dir = os.path.join(CORE_SET_BASE_DIR, args.dataset_name, args.core_set_model_name,
                                    str(args.core_set_size), args.core_set_method)
        if not os.path.isdir(core_set_dir):
            os.makedirs(core_set_dir)
            print("creating core set directory " + core_set_dir)
        return core_set_dir


def get_hidden_states_from_tokens(model, tokenized_dataset, args):
    input_ids = tokenized_dataset['tokenized_input']['input_ids']
    attention_mask = tokenized_dataset['tokenized_input']['attention_mask']

    batch_size = args.core_set_batch_size
    num_inputs = len(input_ids)

    hidden_states = []

    inputs_processed = 0
    while inputs_processed < num_inputs:
        batch_end_idx = min(inputs_processed + batch_size, num_inputs)
        batch_hidden_states = model(input_ids=torch.tensor(input_ids[inputs_processed: batch_end_idx], device=args.device),
                              attention_mask=torch.tensor(attention_mask[inputs_processed: batch_end_idx], device=args.device))
        hidden_states.extend(batch_hidden_states.pooler_output.tolist())
        inputs_processed = batch_end_idx

    return hidden_states


def get_hidden_states(dataset, ratio, args):
    model, tokenizer = get_model_and_tokenizer_from_args(args)
    tokenized_dataset = tokenize(dataset['train'], tokenizer, args.dataset_name, ratio=ratio)
    hidden_states = get_hidden_states_from_tokens(model, tokenized_dataset, args)

    return hidden_states


def random_core_set(dataset, args):
    dataset['train'] = dataset['train'].shuffle(seed=42)
    dataset['train'] = dataset['train'].select(range(int(dataset.num_rows['train'] * args.core_set_size)))

    return dataset


def greedy_k_means_core_set(dataset, args):
    core_set_dir = get_core_set_dir_from_args(args)
    if any(os.scandir(core_set_dir)):
        dataset['train'] = datasets.load_from_disk(core_set_dir)
    else:
        ratio = 0.005
        hidden_states = get_hidden_states(dataset, ratio, args)
        method = GreedyKMeans(hidden_states)
        core_set_idx, cover_distance = method.sample([], int(len(dataset['train']) * ratio * args.core_set_size))

        dataset['train'] = dataset['train'].select(core_set_idx)
        dataset['train'].flatten_indices()
        try:
            dataset['train'].save_to_disk(core_set_dir)
        except Exception as e:
            print(e)

    return dataset

def find_core_set(dataset, args):
    if args.core_set_method == 'whole_set':
        pass
    elif args.core_set_method == 'random':
        dataset = greedy_k_means_core_set(dataset, args)
    elif args.core_set_method == 'greedy_k_means':
        dataset = greedy_k_means_core_set(dataset, args)
    else:
        raise ValueError('no such core_set_method: {}'.format(args.core_set_method))

    return dataset
