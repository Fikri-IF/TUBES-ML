import numpy as np
import util

class Layer:
    def __init__(self, neuronTotal, activationFunction, bias):
        self.neuronTotal = neuronTotal
        if (activationFunction is not None):
            self.activationFunction = activationFunction.lower()
        else:
            self.activationFunction = activationFunction
        self.weights = None
        self.bias = bias
        self.output = None

    def setWeights(self, weights):
        self.weights = np.array(weights)

    def setOutput(self, input, isInputLayer = False):
        self.input = input
        if (isInputLayer):
            self.output = input
        else:
            if (self.activationFunction == "linear"):
                self.output = util.linear(input)
            elif (self.activationFunction == "sigmoid"):
                self.output = util.sigmoid(input)
            elif (self.activationFunction == "relu"):
                self.output = util.relu(input)
            elif (self.activationFunction == "softmax"):
                self.output = util.softmax(input)