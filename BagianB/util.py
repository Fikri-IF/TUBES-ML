import math
import numpy as np

def linear(net):
    return net

def difLinear(net):
    return 1

def sigmoid(net):
    res = np.zeros_like(net)
    for index, value in np.ndenumerate(net):
        res[index]=1/(1+math.exp(-value))
    return res

def difSigmoid(net):
    res = np.zeros_like(net)
    for index, value in np.ndenumerate(net):
        res[index] = (1/(1+math.exp(-value))) * (1 - (1/(1+math.exp(-value))))
    return res

def relu(net):
    res = np.zeros_like(net)
    for index, value in np.ndenumerate(net):
        res[index] = max(0, value)
    return res

def difRelu(net):
    res = np.zeros_like(net)

    for index, value in np.ndenumerate(net):
        if (value < 0):
            res[index] = 0
        else:
            res[index]=1
    return res

def softmax(net):
    res = np.zeros_like(net)
    sum = 0
    for index, value in np.ndenumerate(net):
        sum += math.exp(value)
    for index, value in np.ndenumerate(net):
        res[index] (math.exp(value) / sum)
    return res

def difSoftmax(net):
    sum=0
    for i in net:
        sum+= np.exp(i)/np.sum(np.exp(i)*(1-np.exp(i)/np.sum(np.exp(i))))
    return sum

def sse(t, output):
    sum = 0
    for t_i, output_i in zip(t, output):
        sum += (t_i - output_i)**2
    return sum

def difSse(t,output):
    # print("kepo",t)
    # print("kepo0",output)
    # print("kepo",t.shape[0],output.shape[0])
    return t-output

def cross_entropy(t,output):
    sum=0
    for t_i, output_i in zip(t, output):
            sum += (t_i * np.log(output_i))
    return sum

def difCrossEntropy(t,output):
    return -(t/output)+(1-t)/(1-output)