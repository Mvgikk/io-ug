import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import History, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array, load_img

FAST_RUN = False
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

# b) Załaduj bazę danych i dokonaj jej obróbki

# Ścieżka do danych (folder zawierający obrazy kotów i psów)
base_dir = 'dogs-cats-mini/'

# Listy do przechowywania nazw plików
filenames = os.listdir(base_dir)

# Wyciągnij klasy cat/dog z nazwy pliku
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)  # 1 dla psa
    else:
        categories.append(0)  # 0 dla kota

# Stwórz DataFrame
df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

# Listy do przechowywania danych i etykiet
images = []
labels = []

# Przetworz obrazki na tablice
for index, row in df.iterrows():
    img_path = os.path.join(base_dir, row['filename'])
    img = load_img(img_path, target_size=IMAGE_SIZE)  # Poprawiony rozmiar obrazu do 128x128
    img_array = img_to_array(img) / 255.0  # Normalizacja do zakresu [0, 1]
    images.append(img_array)
    labels.append(row['category'])

# Konwersja list na tablice numpy
images = np.array(images)
labels = np.array(labels)

# Konwersja etykiet do one-hot encoded
labels = to_categorical(labels, num_classes=2)

# Podział danych na zbiór treningowy i walidacyjny
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Data Augmentation dla zbioru treningowego
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Przetworz zbiór walidacyjny
validation_datagen = ImageDataGenerator()

# Generatory danych
train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
validation_generator = validation_datagen.flow(X_val, y_val, batch_size=32)

# c) Skonstruuj, wytrenuj model sieci konwolucyjnej

# Definicja modelu
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))  # 2 because we have cat and dog classes

# Kompilacja modelu
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Podsumowanie modelu
model.summary()

# Zapis modelu podczas trenowania
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# Trenowanie modelu
history = History()
history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // 32,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=len(X_val) // 32,
    callbacks=[history, checkpoint]
)
np.save('history.npy', history.history)


# d) Dokonaj walidacji i podaj krzywą uczenia się

# Wykres krzywej uczenia się dla dokładności
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.grid(True, linestyle='--', color='grey')

# Wykres krzywej uczenia się dla straty
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.grid(True, linestyle='--', color='grey')

plt.tight_layout()
plt.show()
