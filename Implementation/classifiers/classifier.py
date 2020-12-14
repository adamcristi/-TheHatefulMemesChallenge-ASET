import abc


class Classifier:

    def __init__(self):
        pass

    @abc.abstractmethod
    def load_data(self, args):
        pass

    @abc.abstractmethod
    def preprocess(self, *args):
        pass

    @abc.abstractmethod
    def train(self, args):
        pass

    @abc.abstractmethod
    def show_best_result(self, args) -> float:
        pass
