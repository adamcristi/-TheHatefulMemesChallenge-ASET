import pandas as pd
import aspectlib, aspectlib.debug
from Implementation.data.aspects_dataset import transform_data_from_dataframe_to_dict_for_texts, \
    transform_data_from_dataframe_to_list_for_images


class Dataset:

    def __init__(self):
        self.train_dataset = pd.read_json(path_or_buf='/home/cristian_adam/Documents/Github/TheHatefulMemesChallenge-ASET/Implementation/data/data/train.jsonl', lines=True)
        self.dev_seen_dataset = pd.read_json(path_or_buf='/home/cristian_adam/Documents/Github/TheHatefulMemesChallenge-ASET/Implementation/data/data/dev_seen.jsonl', lines=True)
        self.dev_unseen_dataset = pd.read_json(path_or_buf='/home/cristian_adam/Documents/Github/TheHatefulMemesChallenge-ASET/Implementation/data/data/dev_unseen.jsonl', lines=True)
        self.test_seen_dataset = pd.read_json(path_or_buf='/home/cristian_adam/Documents/Github/TheHatefulMemesChallenge-ASET/Implementation/data/data/test_seen.jsonl', lines=True)
        self.test_unseen_dataset = pd.read_json(path_or_buf='/home/cristian_adam/Documents/Github/TheHatefulMemesChallenge-ASET/Implementation/data/data/test_unseen.jsonl', lines=True)

        #self.train_dataset = pd.read_json(path_or_buf='./data/train.jsonl', lines=True)
        #self.dev_seen_dataset = pd.read_json(path_or_buf='./data/dev_seen.jsonl', lines=True)
        #self.dev_unseen_dataset = pd.read_json(path_or_buf='./data/dev_unseen.jsonl', lines=True)
        #self.test_seen_dataset = pd.read_json(path_or_buf='./data/test_seen.jsonl', lines=True)
        #self.test_unseen_dataset = pd.read_json(path_or_buf='./data/test_unseen.jsonl', lines=True)

    @transform_data_from_dataframe_to_dict_for_texts
    @aspectlib.debug.log(print_to=open('./logs_dataset.txt', 'a+'), use_logging=None)
    def get_texts_from_train_dataset(self):
        return self.train_dataset

    @transform_data_from_dataframe_to_dict_for_texts
    @aspectlib.debug.log(print_to=open('./logs_dataset.txt', 'a+'), use_logging=None)
    def get_texts_from_dev_seen_dataset(self):
        return self.dev_seen_dataset

    @transform_data_from_dataframe_to_dict_for_texts
    @aspectlib.debug.log(print_to=open('./logs_dataset.txt', 'a+'), use_logging=None)
    def get_texts_from_dev_unseen_dataset(self):
        return self.dev_unseen_dataset

    @transform_data_from_dataframe_to_dict_for_texts
    @aspectlib.debug.log(print_to=open('./logs_dataset.txt', 'a+'), use_logging=None)
    def get_texts_from_test_seen_dataset(self):
        return self.test_seen_dataset

    @transform_data_from_dataframe_to_dict_for_texts
    @aspectlib.debug.log(print_to=open('./logs_dataset.txt', 'a+'), use_logging=None)
    def get_texts_from_test_unseen_dataset(self):
        return self.test_unseen_dataset

    @transform_data_from_dataframe_to_list_for_images
    @aspectlib.debug.log(print_to=open('./logs_dataset.txt', 'a+'), use_logging=None)
    def get_images_from_train_dataset(self):
        return self.train_dataset

    @transform_data_from_dataframe_to_list_for_images
    @aspectlib.debug.log(print_to=open('./logs_dataset.txt', 'a+'), use_logging=None)
    def get_images_from_dev_seen_dataset(self):
        return self.dev_seen_dataset

    @transform_data_from_dataframe_to_list_for_images
    @aspectlib.debug.log(print_to=open('./logs_dataset.txt', 'a+'), use_logging=None)
    def get_images_from_dev_unseen_dataset(self):
        return self.dev_unseen_dataset

    @transform_data_from_dataframe_to_list_for_images
    @aspectlib.debug.log(print_to=open('./logs_dataset.txt', 'a+'), use_logging=None)
    def get_images_from_test_seen_dataset(self):
        return self.test_seen_dataset

    @transform_data_from_dataframe_to_list_for_images
    @aspectlib.debug.log(print_to=open('./logs_dataset.txt', 'a+'), use_logging=None)
    def get_images_from_test_unseen_dataset(self):
        return self.test_unseen_dataset


#if __name__ == "__main__":
#
#    obj = Dataset()
#    texts_data = [obj.get_texts_from_train_dataset(),
#                  obj.get_texts_from_dev_seen_dataset(), obj.get_texts_from_dev_unseen_dataset(),
#                  obj.get_texts_from_test_seen_dataset(), obj.get_texts_from_test_unseen_dataset()]
#
#    images_data = [obj.get_images_from_train_dataset(),
#                   obj.get_images_from_dev_seen_dataset(), obj.get_images_from_dev_unseen_dataset(),
#                   obj.get_images_from_test_seen_dataset(), obj.get_images_from_test_unseen_dataset()]

