import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.applications import VGG16 # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.losses import SparseCategoricalCrossentropy # type: ignore
from sklearn.neighbors import NearestNeighbors

def preprocess_image(image, label):
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def preprocess_datasets(train_dataset, test_dataset):
    train_dataset = train_dataset.map(preprocess_image).shuffle(1000).batch(32)
    test_dataset = test_dataset.map(preprocess_image).batch(32)
    return train_dataset, test_dataset

def load_dataset(data_dir):
    (train_dataset, test_dataset), dataset_info = tfds.load(
        name='caltech101',
        split=['train[:80%]', 'test[20%:]'],
        with_info=True,
        as_supervised=True,
        data_dir=data_dir,
        download=False  # No volver a descargar
    )
    num_classes = dataset_info.features['label'].num_classes
    return (train_dataset, test_dataset), dataset_info, num_classes

# Función para extraer características y guardar imágenes originales
def extract_features(dataset):
    # Cargar el modelo VGG16 con pesos preentrenados de ImageNet, sin la capa de clasificación superior
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Crear un nuevo modelo que produzca los mapas de características
    model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)
    
    features = []
    labels = []
    img = []  # Lista para almacenar las imágenes originales
    
    for images, lbls in dataset:
        # Obtener los mapas de características
        feature_maps = model.predict(images)
        features.append(feature_maps)
        labels.append(lbls.numpy())
        
        # Guardar las imágenes originales
        img.append(images.numpy())
    
    # Convertir las listas en arrays de numpy
    features = np.concatenate(features)
    labels = np.concatenate(labels)
    img = np.concatenate(img)
    
    return features, labels, img

def save_extracted_data(features, labels, images, prefix):
    # Guardar las características, etiquetas e imágenes extraídas
    np.save(f'{prefix}_features.npy', features)
    np.save(f'{prefix}_labels.npy', labels)
    np.save(f'{prefix}_images.npy', images)
    

def load_extracted_data(prefix):
    features = np.load(f'{prefix}_features.npy')
    labels = np.load(f'{prefix}_labels.npy')
    images = np.load(f'{prefix}_images.npy')
    return features, labels, images

def setup_knn_index():
    train_features_flat = np.load('train_images_reshaped.npy')
    knn = NearestNeighbors(n_neighbors=5, algorithm='auto')
    knn.fit(train_features_flat)
    return knn

def search_similar_images(image_feature, knn):
    image_feature_flat = image_feature.reshape((1, -1))
    distances, indices = knn.kneighbors(image_feature_flat)
    return indices, distances

def load_images(prefix):
    return np.load(f'{prefix}_images.npy')
