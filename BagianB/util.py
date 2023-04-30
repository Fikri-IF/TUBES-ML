import math

def difLinear(net):
    return 1

def difSigmoid(net):
    res = []
    for i in net:
        res.append((1/(1+math.exp(-i))) * (1 - (1/(1+math.exp(-i)))))

def difReLU(net):
    res = []
    for i in net:
        if (i < 0):
            res.append(0)
        else:
            res.append(1)

def sse(t, output):
    sum = 0
    for t_i, output_i in zip(t, output):
        sum += (t_i - output_i)**2
    return sum