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
            self.layers[-1].setWeights(weight)
    
    def forwardPropagation(self, input):
        pred = []
        for i in range(len(self.layers)):
            if (i == 0):
                print("INPUT AWAL")
                print(input)
                self.layers[i].setOutput(input, True)
                continue
            print("ITERASI", i)
            print("\nOUTPUT SEBELUMNYA")
            print(self.layers[i-1].output)
            print("\nWEIGHT")
            print(self.layers[i].weights)
            print("\nBIAS")
            print(self.layers[i].bias)
            # h_k = f(b_k + W_k * h_k-1)
            result = np.dot(self.layers[i-1].output, self.layers[i].weights)
            print("RESULT\n")
            print(result)
            self.layers[i].setOutput(self.layers[i].bias + result)
            print("\nOUTPUT")
            print(self.layers[i].output)
            print(i)        
        for i in range(len(self.layers[-1].output)):
            pred.append(np.argmax(self.layers[-1].output[i]))
        pred = np.array(pred).reshape(-1, 1)
        return pred
        # pred = np.argmax(self.layers[-1].output, axis = 1) # y = h(l)
        # return np.reshape(pred, (pred.shape[0],1))
    
    def backwardPropagation(self, prediction):
        dE_dOut = 0
        val = 0
        for i in range (len(self.layers)):
            if (i == 0): 
                continue # input layer tidak perlu update bobot

            # perhitungan total error
            
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
                diffOut = util.difSoftmax(self.layers[index].output)
                val = val * diffOut
            elif (self.layers[index].activationFunction=="relu"):
                diffOut = util.difRelu(self.layers[index].output)
                val = val * diffOut
            elif (self.layers[index].activationFunction=="linear"):
                diffOut = util.difLinear(self.layers[index].output)                                        
                val = val * diffOut
            elif (self.layers[index].activationFunction=="sigmoid"):
                diffOut = util.difSigmoid(self.layers[index].output)                                        
                val = val * diffOut
            else:
                quit()
            dInput = self.layers[-2- (len(self.layers)-1-i)].output
            self.layers[i].weights = self.layers[i].weights - (self.learningRate * np.dot(np.array(dInput).T,val))
            self.layers[i].bias  = self.layers[i].bias - (self.learningRate * np.mean(np.dot(np.array(dInput).T, val)))
    
    def train(self, x_train, y_train):
        totalIterations = len(x_train) / self.batchSize
        self.iteration = 0
        err = 99
        while True:
            total_error = 0
            i = 0
            while True:
                if (len(x_train) != self.batchSize):
                    input = x_train[i*self.batchSize:(i+1)*self.batchSize]
                    target = np.array(y_train[i*self.batchSize:(i+1)*self.batchSize]).T
                    # target = np.reshape(target, (target.shape[0], 1))
                else:
                    print("pp",totalIterations)
                    input = x_train[:int(totalIterations)+i]
                    target = np.array(y_train[:int(totalIterations)+i]).T
                   # target = np.reshape(target, (target.shape[0], 1))
                # input = x_train[i*self.batchSize:(i+1)*self.batchSize]
                # target = np.array(y_train[i*self.batchSize:(i+1)*self.batchSize]).T
                
                
               
                print("RESHAPE")
                # print(target)

                prediction = self.forwardPropagation(input)
                self.backwardPropagation(prediction)
                e = np.mean(util.sse(target, prediction)).mean()
                total_error += e
                i += 1
                self.iteration += 1
                if ((self.iteration < self.maxIter) and (i<totalIterations) and (err > self.errorThreshold)):
                    if(self.iteration<self.maxIter):
                        print("Stopped By Max Iterations")
                    if (i<totalIterations):
                        print("Stopped By Batch Size")
                    if (err > self.errorThreshold):
                        print("Stopped By Error Threshold")
                    break
            err = total_error
            if (err > self.errorThreshold):
                break
        