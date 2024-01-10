import os
import cv2
import numpy as np
from PIL import Image
from face_utils import calculate_eye_distance, calculate_nose_width, calculate_jawline_length, calculate_eyebrow_height, calculate_lip_fullness
import dlib


predictor_path = "shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(predictor_path)

def preprocess_image(image, size=(224, 224), mean_features=None, std_features=None):
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # Sprawdzenie, czy obraz ma mniej niż 3 kanały 
    if len(image.shape) < 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # Sprawdzenie, czy obraz ma 4 kanały (RGBA) i konwersja do RGB
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

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
    


def load_images_from_folder(folder, size=(224, 224), mean_features=None, std_features=None):
    images = []
    labels = []
    all_features = [] # Lista na cechy antropometryczne dla wszystkich obrazow

    for person_name in os.listdir(folder):
        person_path = os.path.join(folder, person_name)
        if os.path.isdir(person_path):
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                image = Image.open(image_path)
                image = np.array(image)

                processed_image, features = preprocess_image(image, size, mean_features, std_features)
                
                images.append(processed_image)
                labels.append(person_name)
                all_features.append(features)

    return images, labels, all_features

