import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def make_triplets(images, labels, num_triplets=10000):
    """
    Tworzy tripletty obrazów dla zbioru LFW. Każdy triplet składa się z anchora, obrazu pozytywnego i obrazu negatywnego.
    """
    triplets = []
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    num_classes = len(np.unique(labels))
    idx = [np.where(labels == i)[0] for i in range(num_classes)]

    for _ in range(num_triplets):
        # Losowy wybór klasy anchora i obrazu pozytywnego
        class_anchor = np.random.randint(0, num_classes)
        if len(idx[class_anchor]) < 2:
            continue  # Pomijamy klasy z mniej niż dwoma obrazami
        anchor_idx, positive_idx = np.random.choice(idx[class_anchor], 2, replace=False)

        # Losowy wybór obrazu negatywnego z innej klasy
        negative_class = (class_anchor + np.random.randint(1, num_classes)) % num_classes
        negative_idx = np.random.choice(idx[negative_class])

        anchor = images[anchor_idx]
        positive = images[positive_idx]
        negative = images[negative_idx]

        triplets.append([anchor, positive, negative])

        for triplet in triplets:
            for img in triplet:
                assert img.ndim == 3 and img.shape[-1] == 1, "Obraz w tripletach nie ma wymaganego formatu (height, width, channels)"
                
    return np.array(triplets)
    

def prepare_dataset(images, features, labels, num_triplets=10000):
    triplets = []
    triplet_features = []
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    num_classes = len(np.unique(labels))
    idx = [np.where(labels == i)[0] for i in range(num_classes)]

    for _ in range(num_triplets):
        # Losowy wybór klasy anchora i obrazu pozytywnego
        class_anchor = np.random.randint(0, num_classes)
        if len(idx[class_anchor]) < 2:
            continue  # Pomijamy klasy z mniej niż dwoma obrazami
        anchor_idx, positive_idx = np.random.choice(idx[class_anchor], 2, replace=False)

        # Losowy wybór obrazu negatywnego z innej klasy
        negative_class = (class_anchor + np.random.randint(1, num_classes)) % num_classes
        negative_idx = np.random.choice(idx[negative_class])

        # Dodajemy triplet obrazów
        triplets.append([images[anchor_idx], images[positive_idx], images[negative_idx]])
        
        # Dodajemy cechy jako pojedynczy wektor (bez dodatkowego wymiaru)
        anchor_feature = np.array(features[anchor_idx]).flatten()
        positive_feature = np.array(features[positive_idx]).flatten()
        negative_feature = np.array(features[negative_idx]).flatten()
        triplet_features.append([anchor_feature, positive_feature, negative_feature])

    triplets = np.array(triplets)
    triplet_features = np.array(triplet_features)


    return triplets, triplet_features