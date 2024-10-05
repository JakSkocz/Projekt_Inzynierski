import os
import numpy as np
import cv2
from face_utils import calculate_eye_distance, calculate_nose_width, calculate_jawline_length, calculate_eyebrow_height, calculate_lip_fullness
import dlib
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
from train_model import triplet_loss
from load_data import load_images_from_folder
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog

# Załaduj wytrenowany model
model = load_model('trained_anthropometric_model_v2.h5', custom_objects={'loss': triplet_loss})

# Stwórz nowy model z pojedynczym wejściem
input_shape = model.input_shape[0][1:]  # Kształt obrazu
feature_shape = model.input_shape[3][1]  # Liczba cech antropometrycznych

input_image = Input(shape=input_shape, name="image_input")
input_features = Input(shape=(feature_shape,), name="features_input")
base_model_output = model.layers[3]([input_image, input_features])

single_input_model = Model(inputs=[input_image, input_features], outputs=base_model_output)

folder_path = './lfw_v4'  

# Wczytywanie obrazów i ich przetwarzanie
mean_features = np.load('mean_features.npy')
std_features = np.load('std_features.npy')
images, labels, features = load_images_from_folder(folder_path, mean_features=mean_features, std_features=std_features)

# Utworzenie bazy danych reprezentacji
representations = {}
for processed_image, label, feature in zip(images, labels, features):
    feature = np.squeeze(feature)
    representation = single_input_model.predict([np.array([processed_image]), np.array([feature])])
    # Zapisz tylko wektor cech
    if label not in representations:
        representations[label] = []
    representations[label].append(representation[1].flatten())



predictor_path = "shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(predictor_path)

def preprocess_image(image_path, size=(224, 224), mean_features=None, std_features=None):
     # Wczytaj obraz z dysku
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Jeśli obraz nie zostanie wczytany poprawnie
    if image is None:
        raise ValueError("Nie można wczytać obrazu z:", image_path)

    # Konwersja obrazu do skali szarości, jeśli to konieczne
    if len(image.shape) == 2:  # Obraz w skali szarości
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Dodajemy detekcję twarzy i punktów orientacyjnych
    detected_faces = face_detector(image, 1)
    if not detected_faces:
        raise ValueError("Nie znaleziono twarzy w obrazie")
    
    shape = shape_predictor(image, detected_faces[0])
    landmarks = [(point.x, point.y) for point in shape.parts()]

    # Obliczanie proporcji antropometrycznych 
    eye_distance = calculate_eye_distance(landmarks)
    left_eye_width = np.linalg.norm(np.array(landmarks[36]) - np.array(landmarks[39]))
    right_eye_width = np.linalg.norm(np.array(landmarks[42]) - np.array(landmarks[45]))
    left_eyebrow_width = np.linalg.norm(np.array(landmarks[17]) - np.array(landmarks[21]))
    right_eyebrow_width = np.linalg.norm(np.array(landmarks[22]) - np.array(landmarks[26]))
    nose_width = calculate_nose_width(landmarks)
    nose_length = np.linalg.norm(np.array(landmarks[27]) - np.array(landmarks[33]))
    mouth_width = np.linalg.norm(np.array(landmarks[48]) - np.array(landmarks[54]))
    face_width = np.linalg.norm(np.array(landmarks[0]) - np.array(landmarks[16]))
    face_length = np.linalg.norm(np.array(landmarks[27]) - np.array(landmarks[8]))
    distance_between_inner_eyebrows = np.linalg.norm(np.array(landmarks[21]) - np.array(landmarks[22]))
    distance_between_outer_eyebrows = np.linalg.norm(np.array(landmarks[18]) - np.array(landmarks[25]))
    forehead_length = np.linalg.norm(np.array(landmarks[27]) - np.array(landmarks[24]))
    forehead_height = np.linalg.norm(np.array(landmarks[27]) - np.array(landmarks[21]))
    cheek_to_cheek_distance = np.linalg.norm(np.array(landmarks[1]) - np.array(landmarks[15]))
    cheekbone_width = np.linalg.norm(np.array(landmarks[4]) - np.array(landmarks[12]))
    jawline_length = calculate_jawline_length(landmarks)
    chin_length = np.linalg.norm(np.array(landmarks[57]) - np.array(landmarks[8]))
    upper_lip_height = np.linalg.norm(np.array(landmarks[50]) - np.array(landmarks[62]))
    lower_lip_height = np.linalg.norm(np.array(landmarks[58]) - np.array(landmarks[66]))
    eyebrow_height = calculate_eyebrow_height(landmarks)
    lip_fullness = calculate_lip_fullness(landmarks)

    # Kontynuacja przetwarzania obrazu...
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, size)
    image = image.astype('float32')
    image /= 255.0

    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)

    features = [eye_distance, nose_width, left_eye_width, right_eye_width, left_eyebrow_width, right_eyebrow_width, nose_length, mouth_width, face_width, face_length, distance_between_inner_eyebrows, distance_between_outer_eyebrows, forehead_length, forehead_height, cheek_to_cheek_distance, cheekbone_width, jawline_length, chin_length, upper_lip_height, lower_lip_height, eyebrow_height, lip_fullness]

    # Standaryzacja proporcji
    if mean_features is not None and std_features is not None:
        standardized_features = [np.where(std > 0, (f - mean) / std, 0) for f, mean, std in zip(features, mean_features, std_features)]
        standardized_features = np.array(standardized_features).flatten()
    else:
        standardized_features = np.array(features).flatten()

    return image, standardized_features
 
def format_features(features, indices, feature_names):
    return "\n".join([f"{name}: {features[idx]:.2f}" for name, idx in zip(feature_names, indices)])


def identify_person(image_path, single_input_model, representations, mean_features, std_features):
    new_image, new_features = preprocess_image(image_path, mean_features=mean_features, std_features=std_features)
    new_features = np.squeeze(new_features)
    new_representation = single_input_model.predict([np.array([new_image]), np.array([new_features])])

    # Wybierz tylko drugi element (cechy) z nowej reprezentacji
    new_representation = np.squeeze(np.array(new_representation[1]))

    closest_label = None
    min_distance = float('inf')
    closest_features = None
    for label, reps in representations.items():
        for rep in reps:
            rep = np.squeeze(np.array(rep))
            distance = np.linalg.norm(new_representation - rep)
            if distance < min_distance:
                min_distance = distance
                closest_label = label
                closest_features = rep  

    feature_names = ["Odległość między oczami", "Szerokość nosa", "Długość szczęki", "Wysokość brwi"]
    selected_indices = [0, 1, 16, 20]

    new_features_text = format_features(new_features, selected_indices, feature_names)
    closest_features_text = format_features(closest_features, selected_indices, feature_names)

    # Obliczanie podobieństwa cech antropometrycznych
    max_distance = 100 
    similarity = max(0, (max_distance - min_distance) / max_distance) * 100
    return closest_label, min_distance, closest_features, new_features, similarity, new_features_text, closest_features_text

# GUI
def select_image():
    global panelA, path

    file_path = filedialog.askopenfilename()
    if len(file_path) > 0:
        path = file_path
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))  
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        if panelA is None:
            panelA = tk.Label(image=image)
            panelA.image = image
            panelA.pack(side="left", padx=10, pady=10)
        else:
            panelA.configure(image=image)
            panelA.image = image

def identify_and_show():
    global panelA, path

    if path:
        identified_label, min_distance, closest_features, new_features, similarity, new_features_text, closest_features_text = identify_person(path, single_input_model, representations, mean_features, std_features)

        result_text.set(f"Zidentyfikowana osoba: {identified_label}\n"
                        f"Odległość od najbliższej reprezentacji: {min_distance:.2f}\n"
                        f"Cechy nowego obrazu:\n{new_features_text}\n"
                        f"Cechy najbliższej reprezentacji:\n{closest_features_text}\n"
                        f"Podobieństwo cech antropometrycznych: {similarity:.2f}%")

root = tk.Tk()
root.title("Reidentyfikacja Osób")
root.geometry("300x400") 

panelA = None
path = None  

btn = tk.Button(root, text="Wybierz Obraz", command=select_image)
btn.pack(side="top", fill="both", expand="yes", padx="10", pady="10")

identify_btn = tk.Button(root, text="Identify", command=identify_and_show)
identify_btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

result_text = tk.StringVar()
result_label = tk.Label(root, textvariable=result_text, justify='left')
result_label.pack()

root.mainloop()