import dataset_classes
from model_loader import get_model_and_tokenizer_from_args
from data_tokenizer import tokenize
import torch
import time


def get_hidden_states(model, tokenized_dataset, args):
    input_ids = tokenized_dataset['tokenized_input']['input_ids']
    attention_mask = tokenized_dataset['tokenized_input']['attention_mask']

    batch_size = args.core_set_search_batch_size
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


def find_core_set(dataset, args):
    if args.core_set_method == 'whole_set':
        return dataset
    elif args.core_set_method == 'random':
        dataset['train'] = dataset['train'].shuffle(seed=42)
        dataset['train'] = dataset['train'].select(range(int(dataset.num_rows['train'] * args.core_set_size)))

        # model, tokenizer = get_model_and_tokenizer_from_args(args)
        # TODO: change that, len(dataset) is 3 (inside tokenize)!!
        # tokenized_dataset = tokenize(dataset['train'], tokenizer, args.dataset_name, ratio=1)
        # hidden_states = get_hidden_states(model, tokenized_dataset, args)
        # core_set = dataset['train'].select(range(len(dataset['train'])))

        return dataset
