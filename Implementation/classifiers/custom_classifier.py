from Implementation.classifiers.classifier import *

import numpy as np
import json


class CustomClassifier(Classifier):

    def __init__(self):
        super().__init__()
        self.best_accuracy = 0
        self.data = []

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

    def preprocess(self, args):
        # TODO preprocess the images and texts
        pass

    def train(self, args):
        # TODO create model and train data
        pass

    def show_best_result(self, args) -> float:
        return self.best_accuracy
