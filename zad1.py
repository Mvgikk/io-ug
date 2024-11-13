from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def classify_iris(sl,sw,pl,pw):
    if sl > 4 and pl <= 2 and pw <=0.5:
        return "Setosa"
    elif pl > 5 and pw >= 2:
        return "Virginica"
    else:
        return "Versicolor"
    

df = pd.read_csv('iris.csv')

# print(df)

(train_set, test_set) = train_test_split(df.values, train_size=0.7,random_state=300913)

train_inputs = train_set[:,0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:,0:4]
test_classes = test_set[:,4]

good_predictions= 0
len = test_set.shape[0]

for i in range(len):
    if(classify_iris(sl=test_inputs[i,0], pl= test_inputs[i,2], sw=test_inputs[i,1],pw = test_inputs[i,3]) == test_set[i,4]):
        good_predictions += 1

print(f"good_predictions = {good_predictions}")
print(f"{good_predictions/len*100} %")


train_set = train_set[np.argsort(train_classes)]


# print(train_set)
