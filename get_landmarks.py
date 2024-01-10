import numpy as np
from load_data import load_images_from_folder


# Załaduj dane z pliku

def display_results(images, labels, features, num_samples=5):
    for i in range(num_samples):
        print("Etykieta:", labels[i])
        print("Cechy:", features[i])
        print()

mean_features = np.load('./std_features.npy')
std_features = np.load('./std_features.npy')

images, labels, features = load_images_from_folder('./lfw_v4', mean_features=mean_features, std_features=std_features)
display_results(images, labels, features)
