from ann import ANN as ann
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score, recall_score

dataset = load_iris()

X = dataset.data
Y = dataset.target

xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2)
ann=ann(learningRate=0.1, errorThreshold=0.1, maxIter=200, batchSize=10, countLayer = 4)
ann.addLayer(neuronTotal=3, activationFunction="linear", weight=None, bias=None)
ann.addLayer(neuronTotal=4, activationFunction="linear", weight=None, bias=None)
ann.addLayer(neuronTotal=4, activationFunction="relu",weight=None, bias=None)
ann.addLayer(neuronTotal=4, activationFunction="relu", weight=None, bias=None)
ann.addLayer(neuronTotal=3, activationFunction="linear", weight=None, bias=None)
print("Jumlah layer : " , ann.countLayer)
print("Jumlah iterasi:", ann.iteration)
# print("Jumlah output: ", ann.layers[-1].output.shape[1])

# Print array weight untuk setiap layer
for i in range(1, ann.countLayer):
    print("Layer {} weights".format(i))
    print(ann.layers[i].weights)
    print("")
    print("Layer {} biases".format(i))
    print(ann.layers[i].bias)
    print("")

ann.train(X, Y)
pred = ann.forwardPropagation(X)

accuracy = accuracy_score(Y, pred.flatten())
f1 = f1_score(Y, pred.flatten(), average='weighted')
recall = recall_score(Y, pred.flatten(), average='weighted')

print(f'Accuracy score = {accuracy}')
print(f'Weighted F1 Score = {f1}')
print(f'Recall = {recall}')
accuracy = accuracy_score(ytest, pred)
f1 = f1_score(ytest, pred, average='weighted')
print(f'Accuracy score = {accuracy}')
print(f'F1 Score = {f1}')