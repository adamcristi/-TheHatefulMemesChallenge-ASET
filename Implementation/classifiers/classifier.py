import abc


class Classifier:

    def __init__(self):
        pass

    @abc.abstractmethod
    def load_data(self, args):  # load the images and texts
        pass

    @abc.abstractmethod
    def preprocess(self, *args):  # preprocess the images and texts
        pass

    @abc.abstractmethod
    def train(self, args):  # create model and train data
        pass

    @abc.abstractmethod
    def show_best_result(self, args) -> float:
        pass
