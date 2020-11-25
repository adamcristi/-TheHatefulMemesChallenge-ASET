from Implementation.classifiers.classifier import *
from Implementation.preprocess.images.img2vecencoding import Img2VecEncoding
from Implementation.preprocess.texts.bert_preprocessor import BertPreprocessor

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

        self.data = dict({"id": [], "img": [], "label": [], "text": []})
        for elem in temp_data:
            self.data["id"] += [int(elem["id"])]
            self.data["img"] += [elem["img"]]
            self.data["label"] += [int(elem["label"])]
            self.data["text"] += [elem["text"]]

        self.data["id"] = np.array(self.data["id"])
        self.data["img"] = np.array(self.data["img"])
        self.data["label"] = np.array(self.data["label"])
        self.data["text"] = np.array(self.data["text"])

    def preprocess(self, args=(BertPreprocessor('bert-base-uncased', True), Img2VecEncoding())):

        if type(self.data) is not dict or \
                not all(key in self.data for key in ["id", "label", "text"]) or \
                not all(len(self.data["id"]) == len_other
                        for len_other in [len(self.data["label"]), len(self.data["text"])]):
            raise Exception("Cannot preprocess data with that format!")

        text_preprocessor, image_preprocessor = args

        text_preprocessor.execute(self.data)
        image_preprocessor.execute(self.data)

    def train(self, args):
        # TODO create model and train data
        pass

    def show_best_result(self, args) -> float:
        return self.best_accuracy
