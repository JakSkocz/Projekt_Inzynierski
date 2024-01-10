import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import euclidean_distances

def preprocess_image(image, size=(224, 224)):
    """
    Przetwarza obraz do formatu odpowiedniego dla sieci neuronowej, skaluje i normalizuje.
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, size)
    image = image.astype('float32')
    image /= 255.0
    image = np.expand_dims(image, axis=-1)
    return image

def load_image(image_path, size=(224, 224)):
    """
    Wczytuje i przetwarza pojedyncze zdjęcie.
    """
    img = Image.open(image_path)
    img = preprocess_image(img, size)
    return img

# Zrekonstruuj architekturę modelu
def create_base_network(input_shape):
    input = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)  # Stała liczba cech
    return Model(input, x)

def extract_features(model, image_path):
    image = load_image(image_path)
    image = np.expand_dims(image, axis=0)  # Dodanie wymiaru batch
    return model.predict(image)

def compare_images(model, anchor_image_path, comparison_images_folder):
    anchor_features = extract_features(model, anchor_image_path)
    distances = []
    for img_name in os.listdir(comparison_images_folder):
        img_path = os.path.join(comparison_images_folder, img_name)
        comparison_features = extract_features(model, img_path)
        distance = euclidean_distances(anchor_features, comparison_features)
        distances.append((img_name, distance[0][0]))
    return distances

def create_base_network(input_shape):
    input = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    return Model(input, x)

def main():
    input_shape = (224, 224, 1)
    base_network = create_base_network(input_shape)
    
     # Wczytaj wagi do base_network z pliku
    base_network.load_weights('trained_model_v4.h5', by_name=True)

    anchor_image_path = './lfw_v4/Jakub_Skoczylas/Jakub_Skoczylas_0001.jpg'
    comparison_images_folder = './lfw_v4/Jakub_Skoczylas'

    distances = compare_images(base_network, anchor_image_path, comparison_images_folder)
    for img_name, distance in distances:
        print(f"{img_name}: {distance}")

if __name__ == "__main__":
    main()
