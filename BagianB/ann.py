import numpy as np
from layer import Layer
import util

class ANN:
    def __init__(self, learningRate, errorThreshold, maxIter, batchSize):
        self.learningRate = learningRate
        self.layers = []
        self.errorThreshold = errorThreshold
        self.maxIter = maxIter
        self.batchSize = batchSize

    def addLayer(self, neuronTotal, activationFunction):
        self.layers.append(Layer(neuronTotal, activationFunction))

        if (len(self.layers) > 1):
            self.layers[-1].set_weights(self.layers[-1].n_neuron, self.layers[-2].n_neuron)
    
    def forwardPropagation(self, input):
        for i in range(len(self.layers)):
            if (i == 0):
                self.layers[i].setOutput(input, True)
                continue
            self.layers[i].setOutput(self.layers[i].bias + np.dot(self.layers[i].weights, self.layers[i-1].output))

        return self.layers[-1]
