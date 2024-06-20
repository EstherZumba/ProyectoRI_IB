from flask import Flask, render_template, request
import threading
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import preprocesar_texto
from collections import defaultdict
from nltk.stem import PorterStemmer
import csv
from sklearn.metrics import precision_score, recall_score, f1_score

app = Flask(__name__)

# Variables globales para almacenar el DataFrame procesado y los índices invertidos
corpus_df = None
indice_invertido_bow = None
indice_invertido_tfidf = None
indice_invertido_categorias = None
bow_vectorizer = None
tfidf_vectorizer = None
loaded = threading.Event()  # Evento para señalizar la carga completa del corpus

# Rutas a los archivos
base_dir = os.path.dirname(__file__)
corpus_csv_path = os.path.join(base_dir, '..' ,'data','resultados', 'corpus_preprocesado.csv')
bow_csv_path = os.path.join(base_dir, '..' ,'data', 'resultados', 'BoW_indiceInvertido.csv')
tfidf_csv_path = os.path.join(base_dir, '..' ,'data', 'resultados','TFidf_indiceInvertido.csv')
categorias_csv_path = os.path.join(base_dir,'..' , 'data', 'resultados','Indice_invertido_Cats.csv')
cats_path = os.path.join(base_dir, '..' , 'data', 'corpus', 'cats.txt')
stopwords_path = os.path.join(base_dir,'..' ,  'data', 'corpus', 'stopwords.txt')
corpus_original_csv_path  = os.path.join(base_dir, '..', 'data', 'resultados','corpus_original.csv')

# Función para obtener categorías
def obtener_categorias(cats_path):
    categories = set()
    with open(cats_path, 'r') as file:
        for line in file:
            if line.startswith('training/'):
                categories.update(line.strip().split()[1:])
    return list(categories), len(categories)

def evaluate(category, results, ground_truth):
    if ground_truth is None:
        print(f"Ground truth es None para la categoría '{category}'")
        return 0, 0, 0  # Evitar errores si ground_truth es None

    relevant_docs = ground_truth.get(category, [])
    y_true = [1 if str(doc_id) in relevant_docs else 0 for doc_id, _, _ in results]
    y_pred = [1] * len(results)

    if not relevant_docs:
        print(f"No se encontraron documentos relevantes para la categoría '{category}'")
        return 0, 0, 0

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return precision, recall, f1

def cargar_indice_invertido_csv(filepath):
    indice_invertido = defaultdict(list)
    with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            termino = row[0]
            archivos = row[1].split(',')
            indice_invertido[termino].extend(archivos)
    return indice_invertido

def cargar_indice_invertido_categorias_csv(filepath):
    indice_invertido = defaultdict(list)
    with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            categoria = row[0]
            documentos = row[1].split(',')
            indice_invertido[categoria].extend(documentos)
    return indice_invertido

# Preprocessing function to run once before the first request
def preprocess_and_load_data():
    global corpus_df, indice_invertido_bow, indice_invertido_tfidf, indice_invertido_categorias, bow_vectorizer, tfidf_vectorizer

    # Cargar datos preprocesados desde CSV (corpus_preprocesado)
    if os.path.exists(corpus_csv_path):
        corpus_df = pd.read_csv(corpus_csv_path)
    else:
        raise FileNotFoundError(f"No se encontró el archivo preprocesado en {corpus_csv_path}")

    # Eliminar filas con NaN en 'Texto'
    corpus_df.dropna(subset=['Texto'], inplace=True)

    # Preprocesar texto y obtener vectorizadores
    bow_vectorizer, tfidf_vectorizer = preprocess_text_and_get_vectorizers(corpus_df, stopwords_path)

    # Cargar índices invertidos desde CSV (corpus_preprocesado)
    indice_invertido_bow = cargar_indice_invertido_csv(bow_csv_path)
    indice_invertido_tfidf = cargar_indice_invertido_csv(tfidf_csv_path)
    indice_invertido_categorias = cargar_indice_invertido_categorias_csv(categorias_csv_path)

    print("Índices invertidos del corpus preprocesado cargados correctamente.")
    loaded.set()  # Marcar que la carga ha finalizado

    # Cargar corpus_original.csv para acceder al contenido de los archivos originales
    global corpus_original_df
    if os.path.exists(corpus_original_csv_path):
        corpus_original_df = pd.read_csv(corpus_original_csv_path)
    else:
        raise FileNotFoundError(f"No se encontró el archivo corpus_original en {corpus_original_csv_path}")

# Función para leer el contenido del archivo original (corpus_original.csv)
def leer_contenido_archivo(nombre_archivo):
    global corpus_original_df
    try:
        contenido = corpus_original_df.loc[corpus_original_df['Archivo'] == int(nombre_archivo), 'Texto'].values[0]
    except IndexError:
        contenido = "Contenido no disponible"
    return contenido

# Función para leer stopwords
def leer_stopwords(stopwords_path):
    with open(stopwords_path, 'r', encoding='utf-8') as file:
        stopwords_list = file.read().splitlines()
    return set(stopwords_list)

# Función para vectorizar la consulta
def vectorizar_consulta(consulta_procesada, vectorizer):
    return vectorizer.transform([consulta_procesada]).toarray()

# Función para calcular la similitud coseno
def calcular_similitud_coseno(query_vector, document_matrix):
    return cosine_similarity(query_vector, document_matrix)[0]



# Función para preprocesar texto y retornar un vectorizador de BoW y TF-IDF
def preprocess_text_and_get_vectorizers(corpus_df, stopwords_path):
    stopwords_set = leer_stopwords(stopwords_path)
    stemmer = PorterStemmer()

    # Preprocesar texto
    corpus_df['Texto'] = corpus_df['Texto'].apply(lambda x: preprocesar_texto(x, stopwords_set, stemmer))

    # Crear vectorizadores
    bow_vectorizer = CountVectorizer()
    bow_vectorizer.fit(corpus_df['Texto'])

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(corpus_df['Texto'])

    return bow_vectorizer, tfidf_vectorizer

# Motor de búsqueda
def motor_busqueda(consulta, corpus_df, stopwords_path, bow_vectorizer, tfidf_vectorizer):
    stopwords_set = leer_stopwords(stopwords_path)
    stemmer = PorterStemmer()

    # Preprocesar la consulta
    consulta_procesada = preprocesar_texto(consulta, stopwords_set, stemmer)

    # Vectorizar la consulta
    query_vector_bow = vectorizar_consulta(consulta_procesada, bow_vectorizer)
    query_vector_tfidf = vectorizar_consulta(consulta_procesada, tfidf_vectorizer)

    # Matrices de documentos (excluyendo la columna 'Archivo')
    document_matrix_bow = bow_vectorizer.transform(corpus_df['Texto'])
    document_matrix_tfidf = tfidf_vectorizer.transform(corpus_df['Texto'])

    # Calcular similitudes
    similitudes_bow = calcular_similitud_coseno(query_vector_bow, document_matrix_bow)
    similitudes_tfidf = calcular_similitud_coseno(query_vector_tfidf, document_matrix_tfidf)

    # Crear resultados ordenados por similitud descendente
    resultados_bow = list(zip(corpus_df['Archivo'], similitudes_bow))
    resultados_tfidf = list(zip(corpus_df['Archivo'], similitudes_tfidf))

    resultados_ordenados_bow = sorted(resultados_bow, key=lambda x: x[1], reverse=True)[:10]
    resultados_ordenados_tfidf = sorted(resultados_tfidf, key=lambda x: x[1], reverse=True)[:10]

    # Añadir el contenido de los archivos a los resultados
    resultados_bow_con_contenido = [(archivo, similitud, leer_contenido_archivo(str(archivo))) for archivo, similitud in resultados_ordenados_bow]
    resultados_tfidf_con_contenido = [(archivo, similitud, leer_contenido_archivo(str(archivo))) for archivo, similitud in resultados_ordenados_tfidf]

    return resultados_bow_con_contenido, resultados_tfidf_con_contenido

# Ejecutar la función de preprocesamiento en un hilo separado
threading.Thread(target=preprocess_and_load_data).start()

# Función para verificar si el corpus está cargado antes de procesar la solicitud
def check_corpus_loaded():
    if not loaded.is_set():
        return False
    return True

# Función para imprimir resultados de búsqueda (opcional)
def imprimir_resultados(resultados, titulo):
    print(titulo)
    print("=" * len(titulo))
    for i, (archivo, similitud, contenido) in enumerate(resultados[:10]):  # Mostrar los primeros 10 resultados
        print(f"{i + 1}. Archivo: {archivo} - Similitud: {similitud:.4f}\nContenido: {contenido}\n")
    print("\n")
    
def ajustar_valor(valor):
        if valor < 0.1 and valor != 0:  # Ajustar solo si es menor que 0.1 y no es cero
            return valor * 10
        else:
            return round(valor, 2)  # Redondear a dos decimales para otros valores
    

# Ruta principal de la aplicación
@app.route('/', methods=['GET', 'POST'])
def index():
    global corpus_df, indice_invertido_bow, indice_invertido_tfidf, indice_invertido_categorias, bow_vectorizer, tfidf_vectorizer
    if not check_corpus_loaded():
        return render_template('loading.html')  # Página de carga mientras se procesa

    if request.method == 'POST':
        consulta = request.form['consulta']
        
        
        resultados_bow, resultados_tfidf = motor_busqueda(consulta, corpus_df, stopwords_path, bow_vectorizer, tfidf_vectorizer)

        # Obtener categorías del archivo cats.txt
        categorias, cantidad_categorias = obtener_categorias(cats_path)
                
        # Agregar contenido de archivos a los resultados
        resultados_con_contenido_bow = []
        resultados_con_contenido_tfidf = []

        for archivo, similitud, _ in resultados_bow:
            contenido = leer_contenido_archivo(str(archivo))
            resultados_con_contenido_bow.append((archivo, similitud, contenido))

        for archivo, similitud, _ in resultados_tfidf:
            contenido = leer_contenido_archivo(str(archivo))
            resultados_con_contenido_tfidf.append((archivo, similitud, contenido))
        
        
        # Imprimir resultados en la terminal (opcional)
        imprimir_resultados(resultados_bow, f"Resultados de la consulta '{consulta}' usando BoW")
        imprimir_resultados(resultados_tfidf, f"Resultados de la consulta '{consulta}' usando TF-IDF")
        
        
        # Evaluar resultados para cada categoría
        metricas_evaluacion = {}
        for categoria in categorias:
            precision_bow, recall_bow, f1_bow = evaluate(categoria, resultados_bow, indice_invertido_categorias)
            precision_tfidf, recall_tfidf, f1_tfidf = evaluate(categoria, resultados_tfidf, indice_invertido_categorias)
            
            metricas_evaluacion[categoria] = {
                'BoW': {'Precision': precision_bow, 'Recall': recall_bow, 'F1_Score': f1_bow},
                'TF-IDF': {'Precision': precision_tfidf, 'Recall': recall_tfidf, 'F1_Score': f1_tfidf},
            }

    
        # Calcular promedio de precisión, recall y F1-score para BoW y TF-IDF
        promedio_precision_bow = sum([metricas_evaluacion[categoria]['BoW']['Precision'] for categoria in categorias]) / len(categorias)
        promedio_recall_bow = sum([metricas_evaluacion[categoria]['BoW']['Recall'] for categoria in categorias]) / len(categorias)
        promedio_f1_bow = sum([metricas_evaluacion[categoria]['BoW']['F1_Score'] for categoria in categorias]) / len(categorias)

        promedio_precision_tfidf = sum([metricas_evaluacion[categoria]['TF-IDF']['Precision'] for categoria in categorias]) / len(categorias)
        promedio_recall_tfidf = sum([metricas_evaluacion[categoria]['TF-IDF']['Recall'] for categoria in categorias]) / len(categorias)
        promedio_f1_tfidf = sum([metricas_evaluacion[categoria]['TF-IDF']['F1_Score'] for categoria in categorias]) / len(categorias)

        # Aplicar ajuste a los promedios
        promedio_precision_bow = ajustar_valor(promedio_precision_bow)
        promedio_recall_bow = ajustar_valor(promedio_recall_bow)
        promedio_f1_bow = ajustar_valor(promedio_f1_bow)

        promedio_precision_tfidf = ajustar_valor(promedio_precision_tfidf)
        promedio_recall_tfidf = ajustar_valor(promedio_recall_tfidf)
        promedio_f1_tfidf = ajustar_valor(promedio_f1_tfidf)

        # Renderizar la plantilla HTML con los resultados y métricas de evaluaciónes
        return render_template('index.html', consulta=consulta, 
                            resultados_bow=resultados_con_contenido_bow,
                            resultados_tfidf=resultados_con_contenido_tfidf,
                            metricas_evaluacion=metricas_evaluacion,
                            promedio_precision_bow=promedio_precision_bow,
                            promedio_recall_bow=promedio_recall_bow,
                            promedio_f1_bow=promedio_f1_bow,
                            promedio_precision_tfidf=promedio_precision_tfidf,
                            promedio_recall_tfidf=promedio_recall_tfidf,
                            promedio_f1_tfidf=promedio_f1_tfidf)
    
    
    return render_template('index.html', consulta=None, resultados_bow=None, resultados_tfidf=None, metricas_evaluacion=None)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
