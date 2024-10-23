from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='FlowerType')
print(X.head())

pca_iris = PCA(n_components=2).fit(iris.data)
print(pca_iris)
print(pca_iris.explained_variance_ratio_)
print(pca_iris.components_)
print(pca_iris.transform(iris.data))

X_pca = pca_iris.transform(iris.data)

colors = ['r', 'g', 'b']
target_names = iris.target_names


plt.figure(figsize=(8, 6))
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, label=target_name)

plt.legend()
plt.title('PCA of Iris dataset')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.grid(True)
plt.show()



# pca_full = PCA().fit(iris.data)

# print(f"$Explained Variance Ratio: {pca_full.explained_variance_ratio_}")

# #pierwszy komponent wyjaśnia 92.46% wariancji danych
# #drugi komponent wyjaśnia 5.31%
# #łącznie 97.77% dlatego wystarczą dwa
