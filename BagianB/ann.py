import numpy as np
from layer import Layer
import util

class ANN:
    def __init__(self, learningRate, errorThreshold, maxIter, batchSize, countLayer):
        self.learningRate = learningRate
        self.countLayer=countLayer
        self.layers = []
        self.iteration= None
        self.errorThreshold = errorThreshold
        self.maxIter = maxIter
        self.batchSize = batchSize

    def addLayer(self, neuronTotal, activationFunction, weight, bias):
        self.layers.append(Layer(neuronTotal, activationFunction, bias))
        if (len(self.layers) > 1):
            # set weight untuk layer selain input layer
            self.bias=bias
            for i in range(1,len(self.layers)):
                self.layers[i].setWeights(weight)
    
    def forwardPropagation(self, input):
        for i in range(len(self.layers)):
            if (i == 0):
                self.layers[i].setOutput(input, True)
                print("layer0",self.layers[0].output)
                continue
            result = np.dot(self.layers[i-1].output, self.layers[i].weights)
            self.layers[i].setOutput(self.layers[i].bias + result)
            print("layer",i,self.layers[i].output)

        pred = np.argmax(self.layers[-1].output, axis = 1)
        return np.reshape(pred, (pred.shape[0],1))
    
    def backwardPropagation(self, prediction):
        dE_dOut = 0
        val = 0
        for i in range (1,len(self.layers)):            
            if (self.layers[i].activationFunction == "softmax"): # softmax menggunakan cross entropy
                dE_dOut = util.difCrossEntropy(prediction, self.layers[-1].output)
                # print("de",dE_dOut)
                val = dE_dOut
            else:
                dE_dOut = util.difSse(prediction, self.layers[-1].output)
                # if (i == 1):
                #     print("de1",dE_dOut)
                val = dE_dOut

            diffOut = 0
            for j in range(len(self.layers)-1-i):
                if (self.layers[1-j].activationFunction == "softmax"):
                    diffOut = util.difSoftmax(self.layers[-1-j].input)
                    val = val * diffOut
                    val = np.dot(val, self.layers[-1-j].weights.T)
                elif (self.layers[1-j].activationFunction == "relu"):
                    diffOut = util.difRelu(self.layers[-1-j].input)
                    val = val * diffOut
                    val = np.dot(val, self.layers[-1-j].weights.T)
                elif (self.layers[1-j].activationFunction == "linear"):
                    diffOut = util.difLinear(self.layers[-1-j].input)                                        
                    val = val * diffOut
                    val = np.dot(val, self.layers[-1-j].weights.T)
                elif (self.layers[1-j].activationFunction == "sigmoid"):
                    diffOut = util.difSigmoid(self.layers[-1-j].input)                                        
                    val = val * diffOut
                    val = np.dot(val, self.layers[-1-j].weights.T)
                else:
                    quit()

            index = -1- (len(self.layers)-1-i)
            if (self.layers[index].activationFunction == "softmax"):
                diffOut = util.difSoftmax(self.layers[index].input)
                val = val * diffOut
            elif (self.layers[index].activationFunction=="relu"):
                diffOut = util.difRelu(self.layers[index].input)
                val = val * diffOut
            elif (self.layers[index].activationFunction=="linear"):
                diffOut = util.difLinear(self.layers[index].input)                                        
                val = val * diffOut
            elif (self.layers[index].activationFunction=="sigmoid"):
                diffOut = util.difSigmoid(self.layers[index].input)                                        
                val = val * diffOut
            else:
                quit()
            dInput = self.layers[-2- (len(self.layers)-1-i)].output
            # self.layers[i].weights=np.insert(self.layers[i].weights,0,self.layers[i].bias,axis=0)
            self.layers[i].weights = self.layers[i].weights - (self.learningRate * np.dot((np.array(dInput)).T,(val)))
            self.layers[i].bias  = self.layers[i].bias - (self.learningRate * np.mean(np.dot((np.array(dInput)).T,(val))))
    def train(self, x_train, y_train):
        cumulativeError = float('inf')
        totalEpoch = int(len(x_train) / self.batchSize)
        for epoch in range(totalEpoch): 
            for batch in range(0, len(x_train), self.batchSize):
                x_batch = x_train[batch:batch+self.batchSize]
                y_batch = y_train[batch:batch+self.batchSize]

                # feed forward
                prediction = self.forwardPropagation(x_batch)

                # backward dan update bobot
                self.backwardPropagation(prediction)

                e = np.mean(util.sse(y_batch, prediction)).mean()
                cumulativeError += e
            if cumulativeError <= self.errorThreshold or epoch == totalEpoch-1 or batch == self.maxIter:
                print("Training stopped at epoch:", epoch+1, "with cumulative error:", cumulativeError)
                break
            cumulativeError = 0.0
        