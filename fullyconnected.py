import numpy as np

class FullyConnectedLayer:

    def __init__(self, weights, bias, input):
        self.weights = weights
        self.bias = bias
        self.in_size, self.out_size = weights.shape
        self.input = input.reshape((1, self.in_size))
        self.output = None


    def dense(self):
        self.output = self.input.dot(self.weights) + self.bias

    def relu(self):
        self.output = self.output * (self.output > 0)

    def softmax(self):
        self.output = np.exp(self.output)
        self.output = self.output / np.sum(self.output)