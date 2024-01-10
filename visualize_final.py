import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


features = np.array([rep.flatten() for reps in representations.values() for rep in reps])
labels = np.array([label for label, reps in representations.items() for _ in reps])

# Redukcja wymiarowości do 2D przy użyciu t-SNE
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(features)

# Wizualizacja
plt.figure(figsize=(10, 8))
for label in np.unique(labels):
    indices = labels == label
    plt.scatter(features_2d[indices, 0], features_2d[indices, 1], label=label, alpha=0.5)
plt.legend()
plt.title("Wizualizacja danych treningowych")
plt.xlabel("Komponent 1")
plt.ylabel("Komponent 2")
plt.show()
