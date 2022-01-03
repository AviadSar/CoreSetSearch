import os
import json


work_envs = ['home', 'hadassah']
dataset_names = ['snli']
core_set_model_names = ['roberta-base']
model_names = ['roberta-base']
core_set_sizes = [1, 0.33]
core_set_methods = ['whole_set', 'random']

for work_env in work_envs:
    for dataset_name in dataset_names:
        for core_set_model_name in core_set_model_names:
            for model_name in model_names:
                for core_set_size in core_set_sizes:
                    for core_set_method in core_set_methods:
                        args_file_dir = os.path.join('trainer_args', work_env, dataset_name, core_set_model_name,
                                                     model_name, str(core_set_size), core_set_method)
                        args_file = 'args.json'

                        data_dict = {
                            "model_name": model_name,
                            "split_ratio": [1, 1, 1],
                            "logging_steps": 10,
                            "num_epochs": 1,

                            "work_env": work_env,
                            "dataset_name": dataset_name,
                            "core_set_method": core_set_method,
                            "core_set_size": core_set_size,
                            "core_set_model_name": core_set_model_name

                        }

                        if not os.path.isdir(args_file_dir):
                            os.makedirs(args_file_dir)
                            print("creating args file directory " + args_file_dir)
                        with open(os.path.join(args_file_dir, args_file), 'w') as json_file:
                            json.dump(data_dict, json_file, indent=4)
