import numpy as np
import util

class Layer:
    def __init__(self, neuronTotal, activationFunction):
        self.neutonTotal = neuronTotal
        self.activationFunction = activationFunction
        self.weights = None
        self.bias = np.full(neuronTotal, 1)
        self.output = None

    def setWeights(self, rows, cols):
        self.weights = np.random.uniform(low = -0.05, high = 0.05, size =(rows, cols))

    def setOutput(self, input, output, isInputLayer = False):
        if (isInputLayer):
            self.output = output
        else:
            if (self.activationFunction == "linear"):
                self.output = util.linear(input)
            elif (self.activationFunction == "sigmoid"):
                self.output = util.sigmoid(input)
            elif (self.activationFunction == "ReLU"):
                self.output = util.relu(input)
            elif (self.activationFunction == "softmax"):
                self.output = util.softmax(input)