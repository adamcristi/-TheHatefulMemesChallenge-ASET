from Implementation.classifiers.classifier import *


class ConcreteClassifier1(Classifier):

    def __init__(self):
        super().__init__()
        self.best_accuracy = 0

    def load_data(self, args):
        # TODO load the images and texts
        pass

    def preprocess(self, args):
        # TODO preprocess the images and texts
        pass

    def train(self, args):
        # TODO create model and train data
        pass

    def show_best_result(self, args) -> float:
        return self.best_accuracy
