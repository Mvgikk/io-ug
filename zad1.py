import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


df = pd.read_csv('titanic.csv')
df.pop(df.columns[0])
# print(df)

# df.to_csv("titanic_modified.csv", index=False)

encoder = LabelEncoder()
df['Class'] = encoder.fit_transform(df['Class'])
df['Sex'] = encoder.fit_transform(df['Sex'])
df['Age'] = encoder.fit_transform(df['Age'])
df['Survived'] = encoder.fit_transform(df['Survived'])


# Konwersja danych na reprezentację "one-hot encoding"
encoded_df = pd.get_dummies(df, columns=["Class", "Sex", "Age", "Survived"])
print(encoded_df.head())

# Uruchomienie algorytmu Apriori
frequent_itemsets = apriori(encoded_df, min_support=0.005, use_colnames=True)

# Liczba unikalnych itemsetów w frequent_itemsets
num_itemsets = frequent_itemsets.shape[0]

# Generowanie reguł asocjacyjnych
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8, num_itemsets=num_itemsets)

# Sortowanie reguł według ufności w kolejności malejącej
sorted_rules = rules.sort_values(by="confidence", ascending=False)

# Filtracja reguł dotyczących przeżywalności
survival_rules = sorted_rules[
    (sorted_rules['consequents'].astype(str).str.contains("Survived")) &
    (sorted_rules['confidence'] >= 0.8)
]

# Wyświetlenie filtrowanych reguł
print("Reguły dotyczące przeżywalności:")
print(survival_rules)


# Generowanie wykresów na podstawie danych z `survival_rules`
# Przekształcenie danych do list
rules_labels = [
    f"{', '.join(list(antecedent))} → {', '.join(list(consequent))}"
    for antecedent, consequent in zip(survival_rules["antecedents"], survival_rules["consequents"])
]
rules_support = survival_rules["support"].tolist()
rules_confidence = survival_rules["confidence"].tolist()


# Wnioski z wykresu wsparcia dla reguł asocjacyjnych dotyczących przeżywalności:
# 1. Reguły związane z brakiem przeżycia (Survived_0) wśród kobiet (Sex_1) w klasie drugiej (Class_2) oraz dzieci (Age_0) w klasie drugiej
#    mają najwyższe wsparcie, co wskazuje na to, że te grupy były licznie reprezentowane w danych i miały niskie szanse na przeżycie.
# 2. Reguły wskazujące na przeżycie (Survived_1) wśród pasażerów pierwszej klasy (Class_1) i mężczyzn (Sex_0) mają znacznie niższe wsparcie,
#    co sugeruje, że te grupy były mniej licznie reprezentowane, ale miały większe szanse na przeżycie.
# 3. Ogólnie, reguły o niższym wsparciu mogą być specyficzne dla mniejszych grup pasażerów, podczas gdy te o wyższym wsparciu dotyczą większych populacji.

# Wykres wsparcia
plt.figure(figsize=(10, 6))
plt.barh(rules_labels, rules_support, color="skyblue")
plt.xlabel("Wsparcie")
plt.title("Wsparcie dla reguł asocjacyjnych dotyczących przeżywalności")
plt.tight_layout()
plt.show()



# Wnioski z wykresu ufności dla reguł asocjacyjnych dotyczących przeżywalności:
# 1. Reguły o najwyższej ufności (bliskiej 1.0) dotyczą pasażerów pierwszej klasy (Class_1) i mężczyzn (Sex_0),
#    co oznacza, że te grupy miały bardzo wysokie prawdopodobieństwo przeżycia (Survived_1), jeśli spełniono warunki.
# 2. Kobiety (Sex_1) w niższych klasach (Class_2 i Class_3) mają wysoką ufność związaną z brakiem przeżycia (Survived_0),
#    co potwierdza, że należenie do tych grup było silnym predyktorem śmierci.
# 3. Reguły związane z dziećmi (Age_0) mają zróżnicowaną ufność w zależności od klasy – w pierwszej klasie (Class_1) dzieci miały większe szanse na przeżycie,
#    podczas gdy w niższych klasach (Class_2 i Class_3) szanse były znacznie mniejsze.
# 4. Ogólnie, wysoka ufność dla reguł wskazuje na bardzo silne zależności między klasą, wiekiem i płcią a szansami na przeżycie.


# Wykres ufności
plt.figure(figsize=(10, 6))
plt.barh(rules_labels, rules_confidence, color="lightgreen")
plt.xlabel("Ufność")
plt.title("Ufność dla reguł asocjacyjnych dotyczących przeżywalności")
plt.tight_layout()
plt.show()