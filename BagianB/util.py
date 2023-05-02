import math
import numpy as np

def linear(net):
    return net

def difLinear(net):
    return 1

def sigmoid(net):
    res = np.empty()
    for i in net:
        res.append(1/(1+math.exp(-i)))
    return res

def difSigmoid(net):
    res = np.empty()
    for i in net:
        res.append((1/(1+math.exp(-i))) * (1 - (1/(1+math.exp(-i)))))

def relu(net):
    res = np.empty()
    for i in net:
        res.append(max(0,i))
    return res

def difRelu(net):
    res = np.empty()
    for i in net:
        if (i < 0):
            res.append(0)
        else:
            res.append(1)

def softmax(net):
    res = np.empty()
    sum = 0
    for i in net:
        sum += math.exp(i)
    for i in net:
        res.append(math.exp(i) / sum)
    return res
def difSoftmax(net):
    res=np.empty()
    sum=0
    for i in net:
        sum+= np.exp(i)/np.sum(np.exp(i)*(1-np.exp(i)/np.sum(np.exp(i))))
    return sum

def sse(t, output):
    sum = 0
    for t_i, output_i in zip(t, output):
        sum += (t_i - output_i)**2
    return sum
def cross_entropy(t,output):
    sum=0;
    for t_i, output_i in zip(t, output):
            sum += (t_i * np.log(output_i))
    return sum
   