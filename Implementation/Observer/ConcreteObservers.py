from Observer import StatisticsObserver

class Classifier(StatisticsObserver):
    def __init__(self):
        super().__init__()

    def load_data(self):
        pass

    def preprocess(self):
        pass

    def train(self):
        pass

    def show_best_results(self):
        return []

class ConcreteClassifier1(Classifier):
    def __init__(self):
        super().__init__()

class ConcreteClassifier2(Classifier):
    def __init__(self):
        super().__init__()

class ConcreteClassifier3(Classifier):
    def __init__(self):
        super().__init__()