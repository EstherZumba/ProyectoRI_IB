import numpy as np
from PIL import Image

def verify_images(images_array, indices, valid_indices, expected_shape):
    """
    Verifica el conjunto de imágenes, los índices y el formato de las imágenes.
    
    Parameters:
        images_array (np.ndarray): Array de imágenes.
        indices (np.ndarray): Índices obtenidos de la búsqueda.
        valid_indices (list): Lista de índices válidos.
        expected_shape (tuple): Forma esperada de las imágenes (alto, ancho, canales).
    """
    # Verificar el tamaño del conjunto de imágenes
    print(f"Total de imágenes en el conjunto: {images_array.shape[0]}")

    # Verificar que los índices sean válidos
    max_index = images_array.shape[0] - 1
    for idx in indices[0][valid_indices]:
        if idx < 0 or idx > max_index:
            print(f"Índice fuera de rango: {idx}")
        else:
            print(f"Índice válido: {idx}")

    # Verificar el formato de las imágenes
    for idx in indices[0][valid_indices]:
        if idx < 0 or idx >= images_array.shape[0]:
            continue
        
        img = images_array[idx]

        # Comprobar la forma de la imagen
        if img.shape != expected_shape:
            print(f"Imagen en índice {idx} tiene un formato incorrecto: {img.shape}. Esperado: {expected_shape}")
        else:
            print(f"Imagen en índice {idx} tiene un formato correcto: {img.shape}")

        # Convertir a una imagen PIL para visualización y guardado
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        img_pil.show(title=f"Imagen en índice {idx}")

# Parámetros de ejemplo
expected_shape = (224, 224, 3)  # Ejemplo: imagen RGB de 224x224 píxeles
indices = np.array([[224, 393, 2301, 2513, 2461, 550, 1133, 2069, 1735, 69, 847, 1221, 26, 144, 1299, 1168, 718]])
valid_indices = list(range(len(indices[0])))

# Simulación de un array de imágenes con la forma (n_imagenes, alto, ancho, canales)
# Sustituye esta línea con el array real
train_images_flat_R2 = np.random.rand(3000, 224, 224, 3)  # Ejemplo de datos

# Ejecutar verificación
verify_images(train_images_flat_R2, indices, valid_indices, expected_shape)
