import torch
import os
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, RobertaModel

MODEL_BASE_DIR = 'trained_models'


def get_model_dir_from_args(args):
    if 'model_name' in args.__dict__.keys():
        model_dir = os.path.join(MODEL_BASE_DIR, args.data_args.dataset_name, args.data_args.core_set_model_name,
                                 args.model_name, str(args.data_args.core_set_size), args.data_args.core_set_method)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
            print("creating model and log directory " + model_dir)
    else:
        raise KeyError('no model directory: "args" contain neither a model_dir, nor a core_set_model_dir')
    return model_dir


def load_pretrained(model, args):
    model_dir = get_model_dir_from_args(args)
    if 'eval' in args.__dict__.keys() and args.eval:
        return model.from_pretrained(model_dir)
    else:
        return model


def get_model_attributes(args):
    if 'model_name' in args.__dict__.keys():
        model_name = args.model_name
    elif 'core_set_model_name' in args.__dict__.keys():
        model_name = args.core_set_model_name
    else:
        raise KeyError('no model to load: "args" contain neither a model_name, nor a core_set_model_name')

    if 'dataset_name' not in args.__dict__.keys():
        if args.data_args.dataset_name == 'snli':
            model_type = 'classification'
        else:
            raise KeyError('no such dataset: {}'.format(args.data_args.dataset_name))
    else:
        model_type = 'no_head'

    if 'device' in args.__dict__.keys():
        device = args.device
    else:
        device = args.data_args.device

    return model_name, model_type, device


def get_model_and_tokenizer_from_args(args):
    model, tokenizer = None, None
    model_name, model_type, device = get_model_attributes(args)

    if 'roberta' in model_name:
        tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
        if model_type == 'classification':
            model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=args.num_labels)
        elif model_type == 'no_head':
            model = RobertaModel.from_pretrained(model_name)

    model = load_pretrained(model, args)
    if model and tokenizer:
        model.to(device)
        return model, tokenizer
    raise Exception('no such model: name "{}"'.format(model_name, model_type))