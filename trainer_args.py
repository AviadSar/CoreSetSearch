import os
from transformers import TrainingArguments
from model_loader import get_model_dir_from_args
from data_args import DataArgs


class TrainerArgs(object):
    def __init__(self, json_data):
        self.model_name = json_data["model_name"]
        self.batch_size = None
        self.split_ratio = json_data["split_ratio"]
        self.logging_steps = json_data["logging_steps"]
        self.num_epochs = json_data["num_epochs"]
        self.num_labels = None
        self.data_args = None
        if "eval" in json_data:
            self.eval = json_data["eval"]
        else:
            self.eval = False
        if "test" in json_data:
            self.test = json_data["test"]
        else:
            self.test = False


def get_training_args(args):
    model_dir = get_model_dir_from_args(args)
    return TrainingArguments(
        output_dir=model_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 4,
        gradient_accumulation_steps=128//args.batch_size,
        eval_accumulation_steps=args.batch_size * 3,
        fp16=True,
        fp16_full_eval=True,
        fp16_backend='apex',
        warmup_steps=0,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=args.logging_steps,
        save_total_limit=1,
        seed=42,
        load_best_model_at_end=True,
        metric_for_best_model='eval_accuracy',
        greater_is_better=True
    )