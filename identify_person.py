import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import os
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageTk
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
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

def extract_features(model, image_path):
    """
    Ekstrahuje cechy z pojedynczego obrazu używając wytrenowanego modelu.
    """
    image = load_image(image_path)
    image = np.expand_dims(image, axis=0)
    return model.predict(image)

def identify_person(model, test_image_path, database_path):
    """
    Identyfikuje osobę na zdjęciu testowym, porównując z obrazami w bazie danych,
    i zwraca również odległość do najbliższego zdjęcia.
    """
    test_features = extract_features(model, test_image_path)
    min_distance = float('inf')
    identified_person = None

    for person_name in os.listdir(database_path):
        person_folder = os.path.join(database_path, person_name)
        if os.path.isdir(person_folder):
            for image_name in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_name)
                comparison_features = extract_features(model, image_path)
                distance = euclidean_distances(test_features, comparison_features)
                if distance < min_distance:
                    min_distance = distance
                    identified_person = person_name

    return identified_person, min_distance


def select_image(root):
    global image_label, result_label
    file_path = filedialog.askopenfilename()
    if file_path:
        # Wyświetlanie wybranego obrazu
        img = Image.open(file_path)
        img = img.resize((224, 224), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=img)
        image_label.config(image=imgtk)
        image_label.image = imgtk

        # Zresetuj wynik i rozpocznij analizę
        result_label.config(text="Analizowanie...")
        root.update_idletasks()  # Aktualizacja interfejsu

        # Wykonywanie predykcji i wyświetlanie wyniku
        identified_person, min_distance = identify_person(base_network, file_path, database_path)
        result_label.config(text=f"Zidentyfikowano: {identified_person}\nOdległość: {min_distance[0][0]}")



def main():
    global base_network, database_path, image_label, result_label
    input_shape = (224, 224, 1)
    base_network = create_base_network(input_shape)
    base_network.load_weights('trained_model_v4.h5', by_name=True)
    database_path = './lfw_v4'  # Ścieżka do bazy danych

    # Interfejs użytkownika
    root = tk.Tk()
    root.title("Identifikacja Osoby")
    root.geometry("500x600")

    browse_button = tk.Button(root, text="Wybierz zdjęcie", command=lambda: select_image(root))
    browse_button.pack()

    image_label = tk.Label(root)
    image_label.pack()

    result_label = tk.Label(root, text="Wynik identyfikacji pojawi się tutaj", justify=tk.LEFT)
    result_label.pack()

    root.mainloop()

if __name__ == "__main__":
    main()