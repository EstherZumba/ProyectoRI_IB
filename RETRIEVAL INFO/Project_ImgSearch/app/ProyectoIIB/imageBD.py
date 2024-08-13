import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory  # type: ignore

def load_dataset(data_dir):
    # Cargar el conjunto de datos desde la carpeta personalizada
    # Utilizar validation_split para dividir el dataset en entrenamiento y prueba
    dataset = image_dataset_from_directory(
        directory=data_dir,
        image_size=(224, 224),
        batch_size=32,
        label_mode='int',  # Puedes cambiar a 'categorical' si prefieres one-hot encoding
        validation_split=0.2,  # 20% para el conjunto de prueba
        subset="training",  # Usar el subconjunto de entrenamiento
        seed=123  # Semilla para reproducibilidad
    )
    test_dataset = image_dataset_from_directory(
        directory=data_dir,
        image_size=(224, 224),
        batch_size=32,
        label_mode='int',  # Puedes cambiar a 'categorical' si prefieres one-hot encoding
        validation_split=0.2,  # 20% para el conjunto de prueba
        subset="validation",  # Usar el subconjunto de prueba
        seed=123  # Semilla para reproducibilidad
    )
    
    return dataset, test_dataset

def get_dataset_size(dataset):
    size = 0
    for batch in dataset:
        size += batch[0].shape[0]  # Tamaño del lote
        # Imprime las primeras 5 imágenes y etiquetas
        if size <= 5:
            images, labels = batch
            print(f"Imagen: {images.numpy().shape}, Etiqueta: {labels.numpy()}")
    return size

if __name__ == "__main__":
    data_dir = 'D:\\SEPTIMO SEMESTRE II\\RI\\KevinMaldonado99\\RETRIEVAL INFO\\Project_ImgSearch\\data\\imagenesOriginales\\101_ObjectCategories'
    
    # Cargar el dataset balanceado
    train_dataset, test_dataset = load_dataset(data_dir)

    # Verificar el tamaño de los conjuntos de datos
    train_size = get_dataset_size(train_dataset)
    test_size = get_dataset_size(test_dataset)

    print(f"Total de imágenes en el dataset de entrenamiento: {train_size}")
    print(f"Total de imágenes en el dataset de prueba: {test_size}")
    print(f"Total de imágenes en el dataset completo: {train_size + test_size}")
