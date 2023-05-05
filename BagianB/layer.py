import numpy as np
import util

class Layer:
    def __init__(self, neuronTotal, activationFunction, weight, bias):
        self.neuronTotal = neuronTotal
        if (activationFunction is not None):
            self.activationFunction = activationFunction.lower()
        else:
            self.activationFunction = activationFunction
        if (weight is not None):
            self.weights = np.array(weight)
        else:
            self.weights = weight
        if (bias is not None):
            self.bias = np.array(bias)
        else:
            self.bias = bias
        self.output = None 

    def setWeights(self, weights):
        self.weights = np.array(weights)
    
    def setWeightsIris(self, rows, cols):
         # assign nilai random dengan range -0.05 sampai dengan 0.05
        self.bias = np.full(self.neuronTotal, 1) # set bias = 1
        self.weights = np.random.uniform(low = -0.05, high = 0.05, size =(rows, cols))

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