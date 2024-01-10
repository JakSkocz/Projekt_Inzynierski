from PIL import Image
import numpy as np
import dlib


def calculate_eye_distance(landmarks):
    left_eye_center = np.array(landmarks[36])  
    right_eye_center = np.array(landmarks[45])
    eye_distance = np.linalg.norm(left_eye_center - right_eye_center)
    return eye_distance

def calculate_nose_width(landmarks):
    left_nostril = np.array(landmarks[31])
    right_nostril = np.array(landmarks[35])
    nose_width = np.linalg.norm(left_nostril - right_nostril)
    return nose_width

def calculate_jawline_length(landmarks):
    left_first = np.array(landmarks[0])
    left_second = np.array(landmarks[4])
    right_first = np.array(landmarks[12])
    right_second = np.array(landmarks[16])
    jawline_length = np.linalg.norm((left_first - left_second) + (right_first - right_second)) / 2
    return jawline_length

def calculate_eyebrow_height(landmarks):
    left_first = np.array(landmarks[19])
    left_second = np.array(landmarks[37])
    right_first = np.array(landmarks[24])
    right_second = np.array(landmarks[44])
    eyebrow_height = np.linalg.norm((left_first - left_second) + (right_first - right_second)) / 2
    return eyebrow_height

def calculate_lip_fullness(landmarks):
    left_first = np.array(landmarks[50])
    left_second = np.array(landmarks[62])
    right_first = np.array(landmarks[58])
    right_second = np.array(landmarks[66])
    lip_fullness = np.linalg.norm((left_first - left_second) + (right_first - right_second)) / 2
    return lip_fullness


# Dopisać pozostałe funkcje z load_data

# Test
#image_path = './lfw_v4/Jakub_Skoczylas/Jakub_Skoczylas_0001_aug_7.jpg'
#landmarks = get_face_landmarks(image_path)
#eye_distance = calculate_eye_distance(landmarks)
#nose_width = calculate_nose_width(landmarks)
#jawline_length = calculate_jawline_length(landmarks)
#print(f"Odległość między oczami: {eye_distance}")
#print(f"Szerokość nosa: {nose_width}")
#print(f"Długość szczęki: {jawline_length}")
#print(landmarks)