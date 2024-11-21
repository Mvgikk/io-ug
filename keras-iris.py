import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Preprocess the data
# Scale the features
# a )standaryzacja danych wokół 0 
#[5.1, 7.0, 6.3] -> [-1.29, 1.09, 0.21],
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode the labels
# b) przekszatałcenie etykiet na macierz binarną
#[0, 1, 2] ->[[1, 0, 0], [0, 1, 0], [0, 0, 1]], 
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

# Define the model
model = Sequential([
    Dense(64, activation='leaky_relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='leaky_relu'),
    Dense(y_encoded.shape[1], activation='softmax')
])
# c) Model ma 4 neurony w warstwie wejsciowej - długosc szerokosc kielicha i platka
# x_train.shape[0] - liczba wierszy x_train.shape[1] - liczba kolumn

# warstwa wyjściowa ma 3 neurony
# y_encoded.shape[1] - liczba kolumn w zakodowanych etykietach


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
# ^ zmieniona szybkość uczenia, 
# rozne optymalizatory daja rozne wyniki sgd troche gorzej wypadl


# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, batch_size=4)

# najlepsza wydajność około 30-40 epoki. po tym stabilnie.
# Brak oznak niedouczenia: Model osiąga wysokie wartości dokładności zarówno na zbiorze treningowym, jak i walidacyjnym, co oznacza, że dobrze nauczył się wzorców z danych.
# Brak oznak przeuczenia: Krzywe walidacyjne i treningowe są blisko siebie i mają stabilny przebieg. Różnica między stratą treningową a walidacyjną nie jest duża, co oznacza, że model generalizuje dobrze.

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")

# Plot the learning curve
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.tight_layout()
plt.show()

# Save the model
model.save('iris_model.h5')

# Plot and save the model architecture
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# h) 
# wczytujemy i przetwarzamy dane standaryzacja cech, one-hot encoding etykiet.
# Następnie dzielimy dane na treningowe i testowe
# Wczytujemy model wytrenowany model iris_model.h5 i trenujemy dalej przez 10 epok.
#   oceniamy nowo wytrenowany model na zbiorze testowym.