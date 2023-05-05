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
        self.layers.append(Layer(neuronTotal, activationFunction, weight, bias))
    
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
        iteration = 0
        cumulativeError = float('inf')
        totalEpoch = int(len(x_train) / self.batchSize)
        print("TOTAL EPOCH", totalEpoch)
        for epoch in range(totalEpoch): 
            for batch in range(0, len(x_train), self.batchSize):
                if (cumulativeError <= self.errorThreshold or batch == self.maxIter or epoch == totalEpoch):
                    if (cumulativeError <= self.errorThreshold):
                        print("Training stopped because cumulative error lower than or equal to error threshold with cumulative error:", cumulativeError)
                    if (epoch == totalEpoch-1):
                        print("Training stopped because epoch is already maximum at epoch:", epoch + 1)
                    if (batch == self.maxIter):
                        print(f"Training stopped because maximum iteration {self.maxIter} is already reached")
                    break

                x_batch = x_train[batch:batch+self.batchSize]
                print()
                y_batch = y_train[batch:batch+self.batchSize]

                # feed forward
                prediction = self.forwardPropagation(x_batch)

                # backward dan update bobot
                self.backwardPropagation(prediction)

                e = np.mean(util.sse(y_batch, prediction)).mean()
                cumulativeError += e
            
            cumulativeError = 0.0
        