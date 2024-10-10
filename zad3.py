import pandas as pd

df = pd.read_csv('iris_with_errors.csv')


#print(df)

# NA, 0(?), - 
print(df.isnull().sum())
