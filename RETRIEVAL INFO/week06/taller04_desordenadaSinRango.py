import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
import pandas as pd

# Descargar los recursos necesarios de NLTK (solo necesita hacerse una vez)
nltk.download('stopwords')
nltk.download('punkt')

# Ruta del archivo
file_path = r'D:\SEPTIMO SEMESTRE II\RI\KevinMaldonado99\RETRIEVAL INFO\week06\corpus\elniñoquesobrevivio.txt'

# Leer el contenido del archivo
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# Limpiar el texto
text = text.lower()

# Tokenizar el texto en palabras
words = nltk.word_tokenize(text)

# Filtrar las stopwords y palabras no alfabéticas
stop_words = set(stopwords.words('spanish'))
filtered_words = [word for word in words if word.isalpha() and word not in stop_words]

# Contar la frecuencia de cada palabra
word_counts = Counter(filtered_words)

# Crear una lista de palabras y sus frecuencias
word_freq = word_counts.items()

# Convertir a DataFrame de pandas
df = pd.DataFrame(word_freq, columns=['Palabra', 'Frecuencia'])

# Guardar el DataFrame en un archivo Excel
excel_file_path = r'D:\SEPTIMO SEMESTRE II\RI\KevinMaldonado99\RETRIEVAL INFO\week06\corpus\frecuencia_palabras_sin_rango.xlsx'
df.to_excel(excel_file_path, index=False)

print(f"Archivo Excel guardado en {excel_file_path}")
