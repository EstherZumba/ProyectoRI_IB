import os
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img #type: ignore

def balancear_imagenes(data_dir, target_count=300):
    # Configuración del generador de imágenes
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Recorrer cada categoría en el directorio
    for category in os.listdir(data_dir):
        category_path = os.path.join(data_dir, category)
        if os.path.isdir(category_path):
            images = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
            image_count = len(images)
            
            if image_count < target_count:
                print(f"Categoría {category}: Aumentando de {image_count} a {target_count} imágenes.")
                required_images = target_count - image_count
                # Generar nuevas imágenes para alcanzar el número deseado
                while required_images > 0:
                    # Seleccionar aleatoriamente una imagen
                    image_name = random.choice(images)
                    img_path = os.path.join(category_path, image_name)
                    img = load_img(img_path)
                    x = img_to_array(img)
                    x = x.reshape((1,) + x.shape)

                    # Generar nuevas imágenes usando el generador
                    i = 0
                    for batch in datagen.flow(x, batch_size=1, save_to_dir=category_path, save_prefix=category, save_format='jpg'):
                        i += 1
                        required_images -= 1
                        if required_images <= 0:
                            break

            elif image_count > target_count:
                print(f"Categoría {category}: Reduciendo de {image_count} a {target_count} imágenes.")
                # Reducción de imágenes
                images_to_remove = random.sample(images, image_count - target_count)
                for image_name in images_to_remove:
                    img_path = os.path.join(category_path, image_name)
                    os.remove(img_path)
                    
            else:
                print(f"Categoría {category}: Ya tiene {target_count} imágenes, no se requiere balanceo.")
        
    print("Proceso de balanceo completado.")

# Ruta del directorio de imágenes
data_dir = 'D:\\SEPTIMO SEMESTRE II\\RI\\KevinMaldonado99\\RETRIEVAL INFO\\Project_ImgSearch\\data\\imagenesOriginales\\101_ObjectCategories'

# Llamar a la función para balancear las imágenes
balancear_imagenes(data_dir)
