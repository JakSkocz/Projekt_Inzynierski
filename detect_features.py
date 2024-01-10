import cv2
import numpy as np
from PIL import Image

def preprocess_image(image, size=(224, 224)):
    """
    Przetwarza obraz do formatu odpowiedniego dla sieci neuronowej, skaluje i normalizuje.
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # Przekształcenie do skali szarości
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image = cv2.resize(image, size)
    image = image.astype('float32')
    image /= 255.0  # Normalizacja

    # Dodanie wymiaru kanału, jeśli jest to konieczne
    if image.ndim == 2:  # Dla obrazów w skali szarości
        image = np.expand_dims(image, axis=-1)

    return image

