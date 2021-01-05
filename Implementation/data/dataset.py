import pandas as pd
import aspectlib, aspectlib.debug
import os
from pathlib import Path
from Implementation.data.aspects_dataset import transform_data_from_dataframe_to_dict_for_texts, \
    transform_data_from_dataframe_to_list_for_images

ROOT_FILENAME = "Implementation"


class Dataset:

    def __init__(self):
        ROOT_DIRECTORY = Path(__file__)
        while str(ROOT_DIRECTORY.name) != ROOT_FILENAME:
            ROOT_DIRECTORY = ROOT_DIRECTORY.parent

        self.train_dataset = pd.read_json(path_or_buf=ROOT_DIRECTORY.joinpath('data', 'data', 'train.jsonl'),
                                          lines=True)
        self.dev_seen_dataset = pd.read_json(path_or_buf=ROOT_DIRECTORY.joinpath('data', 'data', 'dev_seen.jsonl'),
                                             lines=True)
        self.dev_unseen_dataset = pd.read_json(path_or_buf=ROOT_DIRECTORY.joinpath('data', 'data', 'dev_unseen.jsonl'),
                                               lines=True)
        self.test_seen_dataset = pd.read_json(path_or_buf=ROOT_DIRECTORY.joinpath('data', 'data', 'test_seen.jsonl'),
                                              lines=True)
        self.test_unseen_dataset = pd.read_json(
            path_or_buf=ROOT_DIRECTORY.joinpath('data', 'data', 'test_unseen.jsonl'),
            lines=True)

    @transform_data_from_dataframe_to_dict_for_texts
    @aspectlib.debug.log(
        print_to=open(
            os.path.join(str(Path(__file__)).split(ROOT_FILENAME)[0], ROOT_FILENAME, 'data', 'logs_dataset.txt'),
            'a+'), use_logging=None)
    def get_texts_from_train_dataset(self):
        return self.train_dataset

    @transform_data_from_dataframe_to_dict_for_texts
    @aspectlib.debug.log(
        print_to=open(
            os.path.join(str(Path(__file__)).split(ROOT_FILENAME)[0], ROOT_FILENAME, 'data', 'logs_dataset.txt'),
            'a+'), use_logging=None)
    def get_texts_from_dev_seen_dataset(self):
        return self.dev_seen_dataset

    @transform_data_from_dataframe_to_dict_for_texts
    @aspectlib.debug.log(
        print_to=open(
            os.path.join(str(Path(__file__)).split(ROOT_FILENAME)[0], ROOT_FILENAME, 'data', 'logs_dataset.txt'),
            'a+'), use_logging=None)
    def get_texts_from_dev_unseen_dataset(self):
        return self.dev_unseen_dataset

    @transform_data_from_dataframe_to_dict_for_texts
    @aspectlib.debug.log(
        print_to=open(
            os.path.join(str(Path(__file__)).split(ROOT_FILENAME)[0], ROOT_FILENAME, 'data', 'logs_dataset.txt'),
            'a+'), use_logging=None)
    def get_texts_from_test_seen_dataset(self):
        return self.test_seen_dataset

    @transform_data_from_dataframe_to_dict_for_texts
    @aspectlib.debug.log(
        print_to=open(
            os.path.join(str(Path(__file__)).split(ROOT_FILENAME)[0], ROOT_FILENAME, 'data', 'logs_dataset.txt'),
            'a+'), use_logging=None)
    def get_texts_from_test_unseen_dataset(self):
        return self.test_unseen_dataset

    @transform_data_from_dataframe_to_list_for_images
    @aspectlib.debug.log(
        print_to=open(
            os.path.join(str(Path(__file__)).split(ROOT_FILENAME)[0], ROOT_FILENAME, 'data', 'logs_dataset.txt'),
            'a+'), use_logging=None)
    def get_images_from_train_dataset(self):
        return self.train_dataset

    @transform_data_from_dataframe_to_list_for_images
    @aspectlib.debug.log(
        print_to=open(
            os.path.join(str(Path(__file__)).split(ROOT_FILENAME)[0], ROOT_FILENAME, 'data', 'logs_dataset.txt'),
            'a+'), use_logging=None)
    def get_images_from_dev_seen_dataset(self):
        return self.dev_seen_dataset

    @transform_data_from_dataframe_to_list_for_images
    @aspectlib.debug.log(
        print_to=open(
            os.path.join(str(Path(__file__)).split(ROOT_FILENAME)[0], ROOT_FILENAME, 'data', 'logs_dataset.txt'),
            'a+'), use_logging=None)
    def get_images_from_dev_unseen_dataset(self):
        return self.dev_unseen_dataset

    @transform_data_from_dataframe_to_list_for_images
    @aspectlib.debug.log(
        print_to=open(
            os.path.join(str(Path(__file__)).split(ROOT_FILENAME)[0], ROOT_FILENAME, 'data', 'logs_dataset.txt'),
            'a+'), use_logging=None)
    def get_images_from_test_seen_dataset(self):
        return self.test_seen_dataset

    @transform_data_from_dataframe_to_list_for_images
    @aspectlib.debug.log(
        print_to=open(
            os.path.join(str(Path(__file__)).split(ROOT_FILENAME)[0], ROOT_FILENAME, 'data', 'logs_dataset.txt'),
            'a+'), use_logging=None)
    def get_images_from_test_unseen_dataset(self):
        return self.test_unseen_dataset


#if __name__ == "__main__":
#    obj = Dataset()
#    texts_data = [obj.get_texts_from_train_dataset(),
#                  obj.get_texts_from_dev_seen_dataset(), obj.get_texts_from_dev_unseen_dataset(),
#                  obj.get_texts_from_test_seen_dataset(), obj.get_texts_from_test_unseen_dataset()]
#
#    images_data = [obj.get_images_from_train_dataset(),
#                   obj.get_images_from_dev_seen_dataset(), obj.get_images_from_dev_unseen_dataset(),
#                   obj.get_images_from_test_seen_dataset(), obj.get_images_from_test_unseen_dataset()]
