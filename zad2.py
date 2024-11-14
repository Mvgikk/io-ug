import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay




df = pd.read_csv('iris.csv')

# print(df)

(train_set, test_set) = train_test_split(df.values, train_size=0.7,random_state=300913)

train_inputs = train_set[:,0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:,0:4]
test_classes = test_set[:,4]

# print(f"Train inputs \n{train_inputs}")
# print(f"Train classes \n{train_classes}")
# print(f"test inputs \n{test_inputs}")
# print(f"test classes \n{test_classes}")

# print(f"train set \n{train_set}")
# print(f"test set \n{test_set}")

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_inputs,train_classes)


# tree.plot_tree(clf, feature_names=['sepal.length', 'sepal.width', 'petal.length', 'petal.width'], 
#                class_names=clf.classes_)
# plt.show()

predictions = clf.predict(test_inputs)
accuracy = accuracy_score(test_classes, predictions)
print(f"Dokładność DD: {accuracy}")

conf_matrix = confusion_matrix(test_classes, predictions)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=clf.classes_)
disp.plot()
plt.show()