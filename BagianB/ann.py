import numpy as np
from layer import Layer
import util

class ANN:
    def __init__(self, learningRate, errorThreshold, maxIter, batchSize, countLayer):
        self.learningRate = learningRate
        self.countLayer=countLayer;
        self.layers = []
        self.iteration= None;
        self.errorThreshold = errorThreshold
        self.maxIter = maxIter
        self.batchSize = batchSize

    def addLayer(self, neuronTotal, activationFunction):
        self.layers.append(Layer(neuronTotal, activationFunction))
        if (len(self.layers) > 1):
            # set weight untuk layer selain input layer
            self.layers[-1].setWeights(self.layers[-1].neuronTotal, self.layers[-2].neuronTotal)
    
    def forwardPropagation(self, input):
        for i in range(len(self.layers)):
            if (i == 0):
                self.layers[i].setOutput(input, True)
                continue
            # h_k = f(b_k + W_k * h_k-1)
            self.layers[i].setOutput(self.layers[i].bias + np.dot(self.layers[i].weights, self.layers[i-1].output))
        
        return self.layers[-1].output # y = h(l)
    
    def backwardPropagation(self, prediction):
        dE_dOut = 0;
        val = 0;
        for i in range (len(self.layers)):
            if (i == 0): 
                continue # input layer tidak perlu update bobot

            # perhitungan total error
            if (self.layers[i].activationFunction == "softmax"): # softmax menggunakan cross entropy
                dE_dOut = util.cross_entropy(prediction, self.layers[-1].output)
                val = dE_dOut
            else:
                dE_dOut = util.sse(prediction, self.layers[-1].output)
                val = dE_dOut

            diffOut = 0

            for j in range(len(self.layers)-1-i):
                if (self.layers[1-j].activationFunction == "softmax"):
                    diffOut = util.difSoftmax(self.layers[j].output)
                    val = val * diffOut
                elif (self.layers[1-j].activationFunction == "relu"):
                    diffOut = util.difRelu(self.layers[j].output)
                    val = val * diffOut
                elif (self.layers[1-j].activationFunction == "linear"):
                    diffOut = util.difLinear(self.layers[j].output)                                        
                    val = val * diffOut
                elif (self.layers[1-j].activationFunction == "sigmoid"):
                    diffOut = util.difSigmoid(self.layers[j].output)                                        
                    val = val * diffOut
                else:
                    quit()

            if (self.layers[-1-len(self.layers)-1-i].activationFunction == "softmax"):
                diffOut = util.difSoftmax(self.layers[-1-len(self.layers)-1-i].output)
                val = val * diffOut
            elif (self.layers[1-j].activationFunction=="relu"):
                diffOut = util.difRelu(self.layers[-1-len(self.layers)-1-i].output)
                val = val * diffOut
            elif (self.layers[1-j].activationFunction=="linear"):
                diffOut = util.difLinear(self.layers[-1-len(self.layers)-1-i].output)                                        
                val = val * diffOut
            elif (self.layers[1-j].activationFunction=="sigmoid"):
                diffOut = util.difSigmoid(self.layers[-1-len(self.layers)-1-i].output)                                        
                val = val * diffOut
            else:
                quit()
            dInput = self.layers[-2-len(self.layers)-1-i].output
            self.layers[i].weights = self.layers[i].weights - (self.learningRate * np.dot(dInput.T,val))
            self.layers[i].biases  = self.layers[i].bias - (self.learningRate * np.mean(np.dot(dInput.T,value)))
    def train(x_train,y_train):
        epoch = len(x_train)/self.batchSize
        self.iteration=0;
        err=99
        while (err > self.errorThreshold):
            total_error = 0
            i = 0
            while((self.iteration < self.maxIter) and (i<epoch) and (err > self.errorThreshold)):
                input = X[i*self.batchSize:(i+1)*self.batchSize]
                target = np.array(y[i*self.batchSize:(i+1)*self.batchSize]).T
                target = np.reshape(target, (target.shape[0], 1))

                prediction = self.forwardPropagation(input)
                self.backwardPropagation(prediction)
                e = np.mean(util.sse(target, prediction)).mean()
                total_error += e
                i += 1
                self.iteration += 1
            err = total_error
