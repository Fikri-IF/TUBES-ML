from ann import ANN as ann
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score, recall_score

filename = input("Test case file name: ")

def loadJSON(path):
    f = open("test_cases/" + path)
    return json.load(f)

data = loadJSON(filename)
data = data["case"]
dataset = load_iris()

# X = dataset.data
# Y = dataset.target

X = data["input"]
Y = data["target"]
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
ann = ann(learningRate=data["learning_parameters"]["learning_rate"], errorThreshold=data["learning_parameters"]["error_threshold"], maxIter=data["learning_parameters"]["max_iteration"], batchSize=data["learning_parameters"]["batch_size"], countLayer = len(data["model"]["layers"]))
layersCount = len(data["model"]["layers"])
layers = data["model"]["layers"]

ann.addLayer(data["model"]["input_size"], None, np.array(data["initial_weights"][0]), None)
for i, layer in enumerate(layers):
    ann.addLayer(layer["number_of_neurons"], layer["activation_function"], np.array(data["initial_weights"][i][1:]), np.array(data["initial_weights"][i][0]))
print("Jumlah layer : " , ann.countLayer)
print("Jumlah iterasi:", ann.maxIter)

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

# Print array weight untuk setiap layer
for i in range(1, len(ann.layers)):
    print("Layer {} weights".format(i))
    print(ann.layers[i].weights)
    print("")

# accuracy = accuracy_score(np.array(Y).flatten(), pred.flatten())
# f1 = f1_score(np.array(Y).flatten(), pred.flatten(), average='weighted')
# recall = recall_score(np.array(Y).flatten(), pred.flatten(), average='weighted')

# print(f'Accuracy score = {accuracy}')
# print(f'F1 Score = {f1}')
# print(f'Recall = {recall}')
#accuracy = accuracy_score(Y_test, y_pred)
#f1 = f1_score(Y_test, y_pred, average='weighted')

#print(f'Accuracy score = {accuracy}')
#print(f'F1 Score = {f1}')
