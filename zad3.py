import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv')

(train_set, test_set) = train_test_split(df.values, train_size=0.7,random_state=300913)

train_inputs = train_set[:,0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:,0:4]
test_classes = test_set[:,4]


def evaluate_knn(k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_inputs, train_classes)
    
    predictions = knn.predict(test_inputs)
    
    accuracy = accuracy_score(test_classes, predictions)
    conf_matrix = confusion_matrix(test_classes, predictions)
    
    print(f"k-NN (k={k}) - Dokładność: {accuracy}")
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=knn.classes_)
    disp.plot()
    plt.title(f"Macierz błędów dla k-NN (k={k})")
    plt.show()


evaluate_knn(k=3)
evaluate_knn(k=5)
evaluate_knn(k=11)



nb = GaussianNB()
nb.fit(train_inputs, train_classes)

predictions_nb = nb.predict(test_inputs)

accuracy_nb = accuracy_score(test_classes, predictions_nb)
conf_matrix_nb = confusion_matrix(test_classes, predictions_nb)

print(f"Naive Bayes - Dokładność: {accuracy_nb}")
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_nb, display_labels=nb.classes_)
disp.plot()
plt.title("Macierz błędów dla Naive Bayes")
plt.show()
