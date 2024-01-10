import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tensorflow.keras.models import load_model
import cv2
import os
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda



def preprocess_image(image, size=(224, 224)):
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # Sprawdzenie, czy obraz jest już w skali szarości
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image = cv2.resize(image, size)
    image = image.astype('float32')
    image /= 255.0
    image = np.expand_dims(image, axis=-1)
    return image

def create_base_network(input_shape):
    """
    Zrekonstruuj architekturę modelu z oryginalnego skryptu treningowego.
    """
    input = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    return Model(input, x)

def extract_features(model, images):
    features = []
    for img in images:
        feature = model.predict(np.expand_dims(img, axis=0))
        features.append(feature.flatten())
    return np.array(features)



def load_images_from_folder(folder, size=(224, 224)):
    images = []
    labels = []
    for person_name in os.listdir(folder):
        person_path = os.path.join(folder, person_name)
        if os.path.isdir(person_path):
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                img = Image.open(image_path)
                img = preprocess_image(img, size)
                images.append(img)
                labels.append(person_name)
    return images, labels



# Zdefiniowanie ścieżki do zapisanych obrazów i modelu
database_path = './lfw_v4'  # zmień na swoją ścieżkę do danych
model_path = 'trained_model_v4.h5'  # zmień na swoją ścieżkę do modelu

# Rekonstrukcja architektury modelu i załadowanie wag
input_shape = (224, 224, 1)
base_network = create_base_network(input_shape)
base_network.load_weights(model_path, by_name=True)  # zmiana z load_model na load_weights

# Ekstrakcja cech z obrazów
def extract_features(model, images):
    features = []
    for img in images:
        img = preprocess_image(img, size=input_shape[:2])
        feature = model.predict(np.expand_dims(img, axis=0))
        features.append(feature.flatten())
    return np.array(features)

# Załadowanie obrazów i etykiet
images, labels = load_images_from_folder(database_path)

# Ekstrakcja cech
features = extract_features(base_network, images)

# Redukcja wymiarowości do 2D za pomocą t-SNE
tsne = TSNE(n_components=2, random_state=42)
reduced_features = tsne.fit_transform(features)

# Tworzenie wykresu
plt.figure(figsize=(10, 6))
unique_labels = np.unique(labels)
for label in unique_labels:
    indices = [i for i, l in enumerate(labels) if l == label]
    plt.scatter(reduced_features[indices, 0], reduced_features[indices, 1], label=label)
plt.legend()
plt.title('Wizualizacja wektorów cech')
plt.xlabel('Komponent 1')
plt.ylabel('Komponent 2')
plt.show()
