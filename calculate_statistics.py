import os
import numpy as np
from PIL import Image
from get_landmarks import calculate_eye_distance, calculate_nose_width, calculate_jawline_length, calculate_eyebrow_height, calculate_lip_fullness
import dlib
import cv2

predictor_path = "shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(predictor_path)

def preprocess_image(image, size=(224, 224)):
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # Konwersja obrazu do formatu RGB, jeśli jest to konieczne
    if image.shape[2] == 4:  # Dla obrazów RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.ndim == 2:  # Dla obrazów w skali szarości
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Detekcja twarzy i punktów orientacyjnych
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

    return image, [left_eye_width, right_eye_width, left_eyebrow_width, right_eyebrow_width, nose_length, mouth_width, face_width, face_length, distance_between_inner_eyebrows, distance_between_outer_eyebrows, forehead_length, forehead_height, cheek_to_cheek_distance, cheekbone_width, chin_length, upper_lip_height, lower_lip_height, eye_distance, nose_width, eyebrow_height, lip_fullness, jawline_length]

def calculate_features_statistics(folder, size=(224, 224)):
    features_sum = None
    features_sq_sum = None
    count = 0

    for person_name in os.listdir(folder):
        person_path = os.path.join(folder, person_name)
        if os.path.isdir(person_path):
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                image = Image.open(image_path)
                image = np.array(image)
                _, *features = preprocess_image(image, size)

                features = np.array(features)

                if features_sum is None:
                    features_sum = np.zeros(features.shape)
                    features_sq_sum = np.zeros(features.shape)

                features_sum += features
                features_sq_sum += features ** 2
                count += 1

    mean_features = features_sum / count
    std_features = np.sqrt(features_sq_sum / count - mean_features ** 2)

    return mean_features, std_features




folder = './lfw_v4'
mean_features, std_features = calculate_features_statistics(folder)

np.save('mean_features.npy', mean_features)
np.save('std_features.npy', std_features)