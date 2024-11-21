import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Ścieżka do zapisanej historii trenowania i modelu
history_path = 'history.npy'
model_path = 'best_model.keras'

# Załaduj historię trenowania
history = np.load(history_path, allow_pickle=True).item()

# Załaduj zapisany model
model = load_model(model_path)

# Wyświetl podsumowanie modelu
model.summary()

# Wykres krzywej uczenia się dla dokładności
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.grid(True, linestyle='--', color='grey')

# Wykres krzywej uczenia się dla straty
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.grid(True, linestyle='--', color='grey')

plt.tight_layout()
plt.show()
