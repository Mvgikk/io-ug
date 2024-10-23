import pandas as pd

#a

missing_values = ['NA','-']
df = pd.read_csv('iris_with_errors.csv',na_values=missing_values)

print(df.isnull().sum())


#b 


for column in df.columns:
    if column != 'variety':
        
        median_value = df[column].median()

        for index, value in df[column].items():
            if pd.isnull(value) or (value < 0 or value > 15):
                df.at[index, column] = median_value

        print(df[column].values)


#c

valid_records = ['Setosa', 'Versicolor', 'Virginica']

for index, record in df['variety'].items():
    if record not in valid_records:
        print(f"Błędny gatunek {index}: {record}")
        
        prev_record = df.at[index - 1, 'variety'] if index > 0 else None
        next_record = df.at[index + 1, 'variety'] if index < len(df) - 1 else None

        if prev_record == next_record and prev_record in valid_records:
            corrected_record = prev_record
            df.at[index, 'variety'] = corrected_record

        df.at[index, 'variety'] = corrected_record
        print(f"Poprawiono  na: {corrected_record}")

# Wyświetlenie wartości po poprawkach
print("\n variety po poprawkach:")
print(df['variety'].values)