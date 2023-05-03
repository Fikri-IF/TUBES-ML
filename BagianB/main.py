from ann import ANN as ann
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score, recall_score

dataset = load_iris()

X = dataset.data
Y = dataset.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
nn=ann(learningRate=0.1, errorThreshold=0.1, maxIter=200, batchSize=10, countLayer = 4)
nn.addLayer(neuronTotal=4, activationFunction="linear")
nn.addLayer(neuronTotal=4, activationFunction="relu")
nn.addLayer(neuronTotal=4, activationFunction="relu")
nn.addLayer(neuronTotal=3, activationFunction="linear")

print("Jumlah layer : " , nn.countLayer)
print("Jumlah iterasi:", nn.iteration)
# print("Jumlah output: ", nn.layers[-1].output.shape[1])

# Print array weight untuk setiap layer
for i in range(1, nn.countLayer):
    print("Layer {} weights".format(i))
    print(nn.layers[i].weights)
    print("")
    print("Layer {} biases".format(i))
    print(nn.layers[i].bias)
    print("")

nn.train(X, Y)
pred = nn.forwardPropagation(X)

accuracy = accuracy_score(Y, pred.flatten())
f1 = f1_score(Y, pred.flatten(), average='weighted')
recall = recall_score(Y, pred.flatten(), average='weighted')

print(f'Accuracy score = {accuracy}')
print(f'F1 Score = {f1}')
print(f'Recall = {recall}')
#accuracy = accuracy_score(Y_test, y_pred)
#f1 = f1_score(Y_test, y_pred, average='weighted')

#print(f'Accuracy score = {accuracy}')
#print(f'F1 Score = {f1}')