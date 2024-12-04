import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

def process_images(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('jpg'))]
    
    results = []
    
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Nie udało się wczytać obrazu: {image_file}")
            continue

        #    - Dla każdego obrazu wczytujemy go i konwertujemy na skalę szarości.
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 3. **Binaryzacja obrazu**:
        #    - Używamy progu jasności (np. 127) do zamiany obrazu na binaryzację (czarne obiekty na białym tle).
        #    - `THRESH_BINARY_INV` odwraca obraz, aby ptaki były czarne, a tło białe.
        _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)

        # 4. **Znajdowanie konturów**:
        #    - Funkcja `cv2.findContours` znajduje wszystkie kontury na obrazie, które odpowiadają potencjalnym ptakom.
        #    - Każdy kontur traktowany jest jako oddzielny ptak
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # zliczanie ptakow
        bird_count = len(contours)
        results.append((image_file, bird_count))

        # Wyświetlenie przetworzonego obrazu dla wizualizacji
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Oryginalny obraz")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title(f"Progi (ptaki: {bird_count})")
        plt.imshow(binary_image, cmap='gray')
        plt.axis("off")

        plt.tight_layout()
        plt.show()
    
    return results

def count_birds(folder_path):
    print(f"Przetwarzanie obrazów w folderze: {folder_path}")
    results = process_images(folder_path)
    
    for image_file, bird_count in results:
        print(f"{image_file}: {bird_count} ptaków")

folder_path = "resources/bird_miniatures/"
count_birds(folder_path)
