import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('titanic.csv')
df.pop(df.columns[0])
print(df)

# df.to_csv("titanic_modified.csv", index=False)

encoder = LabelEncoder()
df['Class'] = encoder.fit_transform(df['Class'])
df['Sex'] = encoder.fit_transform(df['Sex'])
df['Age'] = encoder.fit_transform(df['Age'])
df['Survived'] = encoder.fit_transform(df['Survived'])