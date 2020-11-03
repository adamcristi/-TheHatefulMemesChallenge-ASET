class Container:
    def __init__(self):
        self.layers = []

    def predict(self, input):
        pass

class Model(Container):
    def __init__(self):
        super.__init__()

    def predict(self, input):
        pass


class SequentialModel(Model):
    def __init__(self):
        super.__init__()

    def predict(self, input):
        for l in self.layers:
            input = l.predict(input)
        return input