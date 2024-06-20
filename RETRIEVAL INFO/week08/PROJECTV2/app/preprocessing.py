import os
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Descargar recursos de NLTK
nltk.download('punkt')

def leer_stopwords(stopwords_path):
    """Lee el archivo de stopwords y devuelve una lista de stopwords."""
    with open(stopwords_path, 'r') as file:
        stopwords_list = file.read().splitlines()
    return set(stopwords_list)

def preprocesar_texto(texto, stopwords_set, stemmer):
    """Limpia y prepara el texto para su análisis."""
    # Tokenizar el texto
    palabras = nltk.word_tokenize(texto)

    # Eliminar caracteres no deseados y normalizar el texto
    palabras_limpias = [palabra.lower() for palabra in palabras if palabra.isalpha()]

    # Eliminar stopwords y aplicar stemming
    palabras_procesadas = [
        stemmer.stem(palabra)
        for palabra in palabras_limpias if palabra.lower() not in stopwords_set
    ]

    # Unir las palabras procesadas en un solo texto
    return ' '.join(palabras_procesadas)

def preprocesar_archivos(training_path, stopwords_path):
    """Preprocesa los archivos en la carpeta /training."""
    stopwords_set = leer_stopwords(stopwords_path)
    stemmer = PorterStemmer()
    data = []

    for root, dirs, files in os.walk(training_path):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                texto = f.read()

            texto_procesado = preprocesar_texto(texto, stopwords_set, stemmer)
            data.append({
                'Archivo': file,
                'Texto': texto_procesado
            })

    corpus_df = pd.DataFrame(data)
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'corpus_preprocesado.csv')
    corpus_df.to_csv(output_path, index=False)
    return corpus_df

def cargar_datos_procesados():
    """Carga los datos preprocesados desde un archivo CSV."""
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'corpus_preprocesado.csv')
    return pd.read_csv(csv_path)
