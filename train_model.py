import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2D, MaxPooling2D, Flatten, concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard
from prepare_data import prepare_dataset
import numpy as np

def triplet_loss(margin=1.0):
    def loss(y_true, y_pred):
        total_length = y_pred.shape[1]
        third_length = total_length // 3

        anchor = y_pred[:, :third_length]
        positive = y_pred[:, third_length:2*third_length]
        negative = y_pred[:, 2*third_length:]

        print("Rozmiar anchor:", anchor.shape)
        print("Rozmiar positive:", positive.shape)
        print("Rozmiar negative:", negative.shape)

        pos_dist = K.sum(K.square(anchor - positive), axis=1)
        neg_dist = K.sum(K.square(anchor - negative), axis=1)
        basic_loss = pos_dist - neg_dist + margin
        return K.maximum(basic_loss, 0.0)

    return loss


def create_base_network(input_shape, feature_shape):
    # Wejście dla obrazów
    input_image = Input(shape=input_shape, name="image_input")
    # Warstwy dla obrazów
    x = Conv2D(64, (3, 3), activation='relu')(input_image)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)

    # Wejście dla cech antropometrycznych
    input_features = Input(shape=(feature_shape,), name="features_input")

    # Połączenie cech obrazu z cechami antropometrycznymi
    combined_features = concatenate([x, input_features], axis=-1)

    # Dodatkowe warstwy po połączeniu
    combined_features = Dense(64, activation='relu')(combined_features)

    return Model(inputs=[input_image, input_features], outputs=combined_features)




def train_model(triplets, triplet_features, epochs=10, batch_size=32):
    input_shape = triplets[0][0].shape 
    feature_shape = triplet_features[0][0].shape[0]  # Liczba cech antropometrycznych

    # Utworzenie podstawowej sieci
    base_network = create_base_network(input_shape, feature_shape)

    # Wejścia dla tripletów
    input_anchor_image = Input(shape=input_shape, name="anchor_image")
    input_positive_image = Input(shape=input_shape, name="positive_image")
    input_negative_image = Input(shape=input_shape, name="negative_image")

    input_anchor_features = Input(shape=(feature_shape,), name="anchor_features")
    input_positive_features = Input(shape=(feature_shape,), name="positive_features")
    input_negative_features = Input(shape=(feature_shape,), name="negative_features")

    # Przetworzenie tripletów przez sieć
    processed_anchor = base_network([input_anchor_image, input_anchor_features])
    processed_positive = base_network([input_positive_image, input_positive_features])
    processed_negative = base_network([input_negative_image, input_negative_features])

    # Połączenie wyników w jedno wyjście
    concatenated = concatenate([processed_anchor, processed_positive, processed_negative], axis=-1)

    # Model końcowy
    model = Model(inputs=[input_anchor_image, input_anchor_features, 
                          input_positive_image, input_positive_features, 
                          input_negative_image, input_negative_features], 
                  outputs=concatenated)

    # Kompilacja modelu
    model.compile(loss=triplet_loss(margin=1.0), optimizer=Adam(learning_rate=0.0001))

    # Przygotowanie danych do treningu
    anchor_images = np.array([triplet[0] for triplet in triplets])
    print(f"Kształt anchor_images: {anchor_images.shape}")
    positive_images = np.array([triplet[1] for triplet in triplets])
    print(f"Kształt positive_images: {positive_images.shape}")
    negative_images = np.array([triplet[2] for triplet in triplets])
    print(f"Kształt negative_images: {negative_images.shape}")

    anchor_features = np.array([features[0] for features in triplet_features])
    print(f"Kształt anchor_features: {anchor_features.shape}")
    positive_features = np.array([features[1] for features in triplet_features])
    print(f"Kształt positive_features: {positive_features.shape}")
    negative_features = np.array([features[2] for features in triplet_features])
    print(f"Kształt negative_features: {negative_features.shape}")


    # Drukowanie kształtów tensorów
    print("Kształt tensora obrazu:", anchor_images.shape)
    print("Kształt tensora cech antropometrycznych:", anchor_features.shape)
    # Trening modelu
    model.fit([anchor_images, anchor_features, 
               positive_images, positive_features, 
               negative_images, negative_features], 
              np.zeros((len(triplets), 1)),  # Dummy Y
              batch_size=batch_size, epochs=epochs, callbacks=[TensorBoard(log_dir='./logs')])

    # Zapisanie wytrenowanego modelu
    model.save('trained_anthropometric_model.h5')
    return model