import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('diabetes.csv')


(train_set, test_set) = train_test_split(data.values, train_size=0.7, random_state=300913)

train_inputs = train_set[:, :-1]
train_classes = train_set[:, -1]


test_inputs = test_set[:, :-1]
test_classes = test_set[:, -1]

encoder = LabelEncoder()
train_classes = encoder.fit_transform(train_classes)
test_classes = encoder.transform(test_classes)

scaler = StandardScaler()
train_inputs = scaler.fit_transform(train_inputs)
test_inputs = scaler.transform(test_inputs)

mlp = MLPClassifier(hidden_layer_sizes=(6, 3),
                    activation='relu',
                    max_iter=500,
                    random_state=300913)

mlp.fit(train_inputs,train_classes)

predictions = mlp.predict(test_inputs)

accuracy = accuracy_score(test_classes, predictions)
print(f"Dokładność: {accuracy}")

conf_matrix = confusion_matrix(test_classes, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=encoder.classes_)
disp.plot()
plt.title("Macierz błędów dla modelu sieci neuronowej")
plt.show()

#FN gorsze