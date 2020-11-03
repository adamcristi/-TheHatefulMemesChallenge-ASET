from Implementation.classifiers.classifier import *


class ConcreteClassifier1(Classifier):

    def __init__(self):
        super().__init__()
        self.best_accuracy = 0

    def load_data(self):
        # TODO load the images and texts
        pass

    def preprocess(self):
        # TODO preprocess the images and texts
        pass

    def train(self):
        # TODO create model and train data
        pass

    def show_best_result(self) -> float:
        return self.best_accuracy
