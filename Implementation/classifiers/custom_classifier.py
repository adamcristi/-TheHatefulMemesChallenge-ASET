import torch
from transformers import BertTokenizer

from Implementation.classifiers.classifier import *

import numpy as np
import json


class CustomClassifier(Classifier):

    def __init__(self):
        super().__init__()
        self.best_accuracy = 0
        self.data = dict()

    def load_data(self, args):
        data_file = args
        temp_data = []

        try:
            with open(data_file, "r") as file:
                for line in file:
                    temp_data += [json.loads(line)]

        except Exception as e:
            raise

        self.data = dict({"id": [], "label": [], "text": []})
        for elem in temp_data:
            self.data["id"] += [int(elem["id"])]
            self.data["label"] += [int(elem["label"])]
            self.data["text"] += [elem["text"]]

        self.data["id"] = np.array(self.data["id"])
        self.data["label"] = np.array(self.data["label"])
        self.data["text"] = np.array(self.data["text"])

    def preprocess(self, args=('bert-base-uncased', True)):

        if type(self.data) is not dict or \
                not all(key in self.data for key in ["id", "label", "text"]) or \
                not all(len(self.data["id"]) == len_other
                        for len_other in [len(self.data["label"]), len(self.data["text"])]):
            raise Exception("Cannot preprocess data with that format!")

        # get tokenizer
        tokenizer = BertTokenizer.from_pretrained(args[0], do_lower_case=args[1])

        # find the maximum token length
        max_len = 0
        for text in self.data["text"]:
            max_len = max(max_len, len(tokenizer.encode(text, add_special_tokens=True)))

        attention_masks = []
        input_ = []

        # tokenize all of the texts
        for text in self.data["text"]:
            encoded = tokenizer.encode_plus(
                # text to encode
                text,
                # adds "[CLS]" and "[SEP]" tokens
                add_special_tokens=True,
                # pad and truncate all texts
                pad_to_max_length=True,
                truncation=True,
                max_length=64,
                # return pytorch tensors
                return_tensors='pt',
                return_attention_mask=True,
            )

            # add the encoded text
            input_.append(encoded['input'])

            # add the attention mask
            attention_masks.append(encoded['attention_mask'])

        # convert lists to tensors
        self.data["attention_masks"] = torch.cat(attention_masks, dim=0)
        self.data["encoded_text"] = torch.cat(input_, dim=0)
        self.data["label"] = torch.tensor(self.data["label"])

    def train(self, args):
        # TODO create model and train data
        pass

    def show_best_result(self, args) -> float:
        return self.best_accuracy
