import os
import numpy as np
from load_data import load_images_from_folder
from prepare_data import prepare_dataset
from train_model import train_model

def main():
    # Ścieżka do folderu z danymi LFW
    folder_path = './lfw_v4'

    # Wczytywanie i przetwarzanie obrazów oraz ekstrakcja cech antropometrycznych
    mean_features = np.load('mean_features.npy')
    std_features = np.load('std_features.npy')
    images, labels, features = load_images_from_folder(folder_path, mean_features=mean_features, std_features=std_features)

    # Przygotowanie datasetu w formie tripletów obrazów i cech
    triplets, triplet_features = prepare_dataset(images, features, labels)

    # Trenowanie modelu
    model = train_model(triplets, triplet_features, epochs=2, batch_size=32)

    print("Model został wytrenowany i zapisany.")

if __name__ == "__main__":
    main()
