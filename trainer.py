import torch
import data_loader
from arg_parser import parse_trainer_args
from trainer_args import get_training_args
from model_loader import get_model_and_tokenizer_from_args
from data_tokenizer import tokenize
from callbacks import StopEachEpochCallback, EvaluateAndSaveCallback, LoggingCallback, EarlyStoppingCallback
from transformers import Trainer
import dataset_classes
import numpy as np
from datasets import load_metric

accuracy_metric = load_metric("accuracy")


def compute_sequence_accuracy(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)


def compute_metrics(args):
    return compute_sequence_accuracy


def load_and_tokenize_dataset(args, tokenizer):
    dataset = data_loader.load(args.data_args)
    # the ratio of train/dev/test sets where 1 is the full size of each the set
    splits_ratio = args.split_ratio
    if args.eval:
        dataset = dataset['validation']
        splits_ratio = [splits_ratio[1]]
    elif args.test:
        dataset = dataset['test']
        splits_ratio = [splits_ratio[2]]
    else:
        dataset = [dataset['train'], dataset['validation'], dataset['test']]

    tokenized_dataset = []
    for split, ratio in zip(dataset, splits_ratio):
        if ratio == 0:
            continue
        tokenized_dataset.append(tokenize(split, tokenizer, args.data_args.dataset_name, ratio))

    dataset = []
    for split in tokenized_dataset:
        dataset.append(dataset_classes.TextDataset(split['tokenized_input'], split['tokenized_target']))

    return dataset


def set_trainer(args):
    model, tokenizer = get_model_and_tokenizer_from_args(args)
    model.resize_token_embeddings(len(tokenizer))

    dataset = load_and_tokenize_dataset(args, tokenizer)
    if len(dataset) > 1:
        train = dataset[0]
        dev = dataset[1]
    else:
        train = None
        dev = dataset[0]

    trainer = Trainer(
        model=model,
        args=get_training_args(args),
        train_dataset=train,
        eval_dataset=dev,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
        compute_metrics=compute_metrics(args)
    )

    return trainer


if __name__ == '__main__':
    args = parse_trainer_args()
    trainer = set_trainer(args)

    if args.eval:
        trainer.evaluate()
    else:
        try:
            trainer.train(resume_from_checkpoint=True)
        except ValueError as e:
            if 'No valid checkpoint' in e.args[0]:
                trainer.train()
            else:
                raise e