import json


class DataArgs(object):
    def __init__(self, json_data):
        self.work_env = json_data["work_env"]
        self.dataset_name = json_data["dataset_name"]
        self.cache_path = None
        self.core_set_method = json_data["core_set_method"]
        self.core_set_size = json_data["core_set_size"]
        self.core_set_model_name = json_data["core_set_model_name"]
        self.core_set_batch_size = None
        self.device = None
