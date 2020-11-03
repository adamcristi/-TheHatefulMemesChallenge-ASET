class Layer:
    def __init__(self):
        self.output = None
        self.weights = None

    def predict(self, input):
        pass

class CoreLayer(Layer):
    def __init__(self):
        super.__init__()
    def predict(self, input):
        pass

class DenseLayer(CoreLayer):
    def __init__(self):
        super.__init__()

    def predict(self, input):
        self.output = input * self.weights
        return self.output