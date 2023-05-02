import numpy as np
import util

class Layer:
    def __init__(self, neuronTotal, activationFunction):
        self.neuronTotal = neuronTotal
        self.activationFunction = activationFunction.lower()
        self.weights = None
        self.bias = np.full(neuronTotal, 1) # set bias = 1
        self.output = None

    def setWeights(self, rows, cols):
        # assign nilai random dengan range -0.05 sampai dengan 0.05
        self.weights = np.random.uniform(low = -0.05, high = 0.05, size =(rows, cols))

    def setOutput(self, input, isInputLayer = False):
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