from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score



iris = load_iris()

datasets = train_test_split(iris.data, iris.target,
                            test_size=0.7,random_state=300913)

train_data, test_data, train_labels, test_labels = datasets

# 0 setosa 1 versicolor 2 virginica

mlp = MLPClassifier(hidden_layer_sizes=(2,), max_iter=3000, random_state=300913)


mlp.fit(train_data, train_labels)

predictions_test = mlp.predict(test_data)
print(f"2 hidden {accuracy_score(predictions_test, test_labels)}")


mlp = MLPClassifier(hidden_layer_sizes=(3,), max_iter=3000, random_state=300913)


mlp.fit(train_data, train_labels)

predictions_test = mlp.predict(test_data)
print(f"3 hidden {accuracy_score(predictions_test, test_labels)}")

mlp = MLPClassifier(hidden_layer_sizes=(3,3), max_iter=3000, random_state=300913)


mlp.fit(train_data, train_labels)

predictions_test = mlp.predict(test_data)
print(f"3,3 hidden {accuracy_score(predictions_test, test_labels)}")