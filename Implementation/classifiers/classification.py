from Implementation.classifiers.classifier import Classifier


class Classification:

    def __init__(self):
        self.classifier_model = None

    def set_classifier_model(self, classifier_model: Classifier):
        self.classifier_model = classifier_model

    def execute_classifier_model(self):
        self.classifier_model.load_data()
        self.classifier_model.preprocess()
        self.classifier_model.train()
        return self.classifier_model.show_best_result()
