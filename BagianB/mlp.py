from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_iris
from sklearn.metrics import f1_score

datasets = load_iris()

X = datasets.data
y = datasets.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

clf = MLPClassifier(random_state=1, max_iter=10).fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("F1 Score")
print(f1_score(y_test,y_pred,average='micro'))