import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'resources/2/wschod.jpg'
# Wczytanie obrazu w kolorze
image = cv2.imread(image_path)
if image is None:
    raise ValueError("Nie można wczytać obrazu. Sprawdź ścieżkę do pliku.")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Metoda 1: Szarość jako średnia (R+G+B)/3
gray_average = np.round(np.mean(image_rgb, axis=2)).astype('uint8')

# Metoda 2: Szarość jako luminancja (0.299*R + 0.587*G + 0.114*B)
gray_luminance = np.round(
    0.299 * image_rgb[:, :, 0] + 0.587 * image_rgb[:, :, 1] + 0.114 * image_rgb[:, :, 2]
).astype('uint8')

# Wyświetlanie obrazów
plt.figure(figsize=(15, 10))

# Oryginalny obraz
plt.subplot(1, 3, 1)
plt.title("Oryginalny obraz (RGB)")
plt.imshow(image_rgb)
plt.axis("off")

# Obraz w skali szarości - średnia
plt.subplot(1, 3, 2)
plt.title("Szarość - Średnia")
plt.imshow(gray_average, cmap='gray')
plt.axis("off")

# Obraz w skali szarości - luminancja
plt.subplot(1, 3, 3)
plt.title("Szarość - Luminancja")
plt.imshow(gray_luminance, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()
