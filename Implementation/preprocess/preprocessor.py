import abc


class Preprocessor:

    def __init__(self):
        pass

    @abc.abstractmethod
    def execute(self, *args):
        pass


