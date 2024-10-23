from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='FlowerType')


colors = ['r', 'g', 'b']
target_names = iris.target_names

plt.figure(figsize=(8, 6))

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X.loc[y == i, 'sepal length (cm)'], X.loc[y == i, 'sepal width (cm)'], 
                color=color, label=target_name)


plt.legend()
plt.title('Original dataset')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.grid(True)
plt.show()


#normalizacja min-max

scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

colors = ['r', 'g', 'b']
target_names = iris.target_names

plt.figure(figsize=(8, 6))

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_scaled.loc[y == i, 'sepal length (cm)'], X_scaled.loc[y == i, 'sepal width (cm)'], 
                color=color, label=target_name)

plt.legend()
plt.title('Min-Max')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.grid(True)
plt.show()

#normalizacja z-score

scaler_zscore = StandardScaler()
X_zscore = pd.DataFrame(scaler_zscore.fit_transform(X), columns=X.columns)

colors = ['r', 'g', 'b']
target_names = iris.target_names

plt.figure(figsize=(8, 6))

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_scaled.loc[y == i, 'sepal length (cm)'], X_scaled.loc[y == i, 'sepal width (cm)'], 
                color=color, label=target_name)

plt.legend()
plt.title('Z-score')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.grid(True)
plt.show()