import numpy as np
from layer import Layer
import util

class ANN:
    def __init__(self, learningRate, errorThreshold, maxIter, batchSize, countLayer, iris = False):
        self.learningRate = learningRate
        self.countLayer=countLayer
        self.layers = []
        self.iteration= None
        self.errorThreshold = errorThreshold
        self.maxIter = maxIter
        self.batchSize = batchSize
        self.iris = iris

    def addLayer(self, neuronTotal, activationFunction, weight, bias):
        if (self.iris):
            self.layers.append(Layer(neuronTotal, activationFunction, weight, bias, True))
            if (len(self.layers) > 1):
                # set weight untuk layer selain input layer
                self.layers[-1].setWeightsIris(self.layers[-2].neuronTotal, self.layers[-1].neuronTotal)
    
        else:
            self.layers.append(Layer(neuronTotal, activationFunction, weight, bias))
    
    def forwardPropagation(self, input):
        for i in range(len(self.layers)):
            if (i == 0):
                self.layers[i].setOutput(input, True)
                continue
            result = np.dot(self.layers[i-1].output, self.layers[i].weights)
            self.layers[i].setOutput(self.layers[i].bias + result)
        pred = np.argmax(self.layers[-1].output, axis = 1)
        return np.reshape(pred, (pred.shape[0],1))
    
    def backwardPropagation(self, prediction):
        dE_dOut = 0
        dE_dW = 0
        for i in range (1, len(self.layers)):            
            if (self.layers[i].activationFunction == "softmax"): # softmax menggunakan cross entropy
                dE_dOut = util.difCrossEntropy(prediction, self.layers[-1].output)
                dE_dW = dE_dOut
            else:
                dE_dOut = util.difSse(prediction, self.layers[-1].output)
                dE_dW = dE_dOut

            dOut_dNet = 0
            for j in range(len(self.layers)-1-i):
                if (self.layers[1-j].activationFunction == "softmax"):
                    dOut_dNet = util.difSoftmax(self.layers[-1-j].input)
                    dE_dW = dE_dW * dOut_dNet
                    dE_dW = np.dot(dE_dW, self.layers[-1-j].weights.T)
                elif (self.layers[1-j].activationFunction == "relu"):
                    dOut_dNet = util.difRelu(self.layers[-1-j].input)
                    dE_dW = dE_dW * dOut_dNet
                    dE_dW = np.dot(dE_dW, self.layers[-1-j].weights.T)
                elif (self.layers[1-j].activationFunction == "linear"):
                    dOut_dNet = util.difLinear(self.layers[-1-j].input)                                        
                    dE_dW = dE_dW * dOut_dNet
                    dE_dW = np.dot(dE_dW, self.layers[-1-j].weights.T)
                elif (self.layers[1-j].activationFunction == "sigmoid"):
                    dOut_dNet = util.difSigmoid(self.layers[-1-j].input)                                        
                    dE_dW = dE_dW * dOut_dNet
                    dE_dW = np.dot(dE_dW, self.layers[-1-j].weights.T)
                else:
                    quit()

            index = -1- (len(self.layers)-1-i)
            if (self.layers[index].activationFunction == "softmax"):
                dOut_dNet = util.difSoftmax(self.layers[index].input)
                dE_dW = dE_dW * dOut_dNet
            elif (self.layers[index].activationFunction=="relu"):
                dOut_dNet = util.difRelu(self.layers[index].input)
                dE_dW = dE_dW * dOut_dNet
            elif (self.layers[index].activationFunction=="linear"):
                dOut_dNet = util.difLinear(self.layers[index].input)                                        
                dE_dW = dE_dW * dOut_dNet
            elif (self.layers[index].activationFunction=="sigmoid"):
                dOut_dNet = util.difSigmoid(self.layers[index].input)                                        
                dE_dW = dE_dW * dOut_dNet
            else:
                quit()
            dNet_dW = self.layers[-2- (len(self.layers)-1-i)].output
            self.layers[i].weights = self.layers[i].weights - (self.learningRate * np.dot((np.array(dNet_dW)).T,(dE_dW)))
            self.layers[i].bias  = self.layers[i].bias - (self.learningRate * np.mean(np.dot((np.array(dNet_dW)).T,(dE_dW))))
    
    def train(self, x_train, y_train):
        cumulativeError = float('inf')
        totalEpoch = int(len(x_train) / self.batchSize)
        print("Total Epoch:", totalEpoch)
        stop = False
        i = 0
        while True:
            if (cumulativeError <= self.errorThreshold or stop):
                break
            if (i == self.maxIter):
                print("Cumulative Error:", cumulativeError)
                print(f"Training stopped because maximum iteration {i} is already reached")
                break
            cumulativeError = 0.0
            for epoch in range(totalEpoch): 
                # print("xtrain",len(x_train))
                # print("BATCH", batch)
                # print("MAX ITERATION", self.maxIter)
                # if (batch == self.maxIter or epoch == totalEpoch):
                #     stop = True
                #     if (epoch == totalEpoch-1):
                #         print("Training stopped because epoch is already maximum at epoch:", epoch + 1)
                #     if (batch == self.maxIter):
                #         print(f"Training stopped because maximum iteration {self.maxIter} is already reached")
                #     break
            
                x_batch = x_train[epoch*self.batchSize : (epoch+1)+self.batchSize]
                y_batch = y_train[epoch*self.batchSize : (epoch+1)+self.batchSize]
                # feed forward
                prediction = self.forwardPropagation(x_batch)
                # backward dan update bobot
                self.backwardPropagation(prediction)
                e = np.mean(util.sse(y_batch, prediction)).mean()
                cumulativeError += e
                if (cumulativeError <= self.errorThreshold):
                    stop = True
                    print("Training stopped because cumulative error lower than or equal to error threshold with cumulative error:", cumulativeError)
                    break  
            i += 1                