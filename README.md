# Trabajo Práctico I - Aprendizaje Automático II

## Integrantes
- **Fontana, Gustavo**
- **Leguiza, Claudia**

---

## Redes Densas y Convolucionales

Este repositorio contiene el desarrollo de tres problemáticas que abordan diferentes aplicaciones de redes neuronales densas y convolucionales, con el objetivo de resolver cada una mediante el uso de estas arquitecturas de aprendizaje profundo.

---

## Objetivos

Los principales objetivos de este trabajo son:

1. **Regresión mediante Redes Neuronales**: Crear un modelo de regresión para predecir el índice de rendimiento académico de estudiantes utilizando redes neuronales densas y un conjunto de características relevantes.

2. **Clasificación de Gestos ("Piedra, Papel o Tijeras")**: Implementar un sistema de clasificación de gestos utilizando MediaPipe para la detección de manos y una red neuronal densa para clasificar los gestos en "piedra", "papel" o "tijeras".

3. **Clasificación de Imágenes mediante CNN**: Construir un modelo de clasificación de imágenes en una de seis categorías predefinidas, utilizando redes neuronales convolucionales (CNN) para lograr un rendimiento robusto en la tarea.

---

## Contenido del Repositorio

1. ### **Carpeta PROBLEMA1**
   - Contiene una notebook que desarrolla el **Problema 1**, incluyendo análisis exploratorio, construcción del modelo de regresión y evaluación de métricas de desempeño.
   - Incluye también el dataset en formato `.csv` con la información de características de los estudiantes.

2. ### **Carpeta PROBLEMA2**
   - Esta carpeta incluye tres scripts que cubren las etapas de grabación del dataset, entrenamiento y prueba del modelo de clasificación de gestos:
     - **record-dataset.py**: Captura gestos desde una cámara, guarda las coordenadas de los puntos de referencia en archivos `.npy` organizados en carpetas por clase.
     - **train-gesture-classifier.py**: Entrena un modelo de red densa basado en los datos de gestos y permite configurar la ruta del dataset y la de salida del modelo.
     - **rock-paper-scissors.py**: Utiliza la cámara para clasificar gestos en tiempo real utilizando el modelo entrenado, mostrando el resultado en pantalla.

3. ### **Carpeta PROBLEMA3**
   - Contiene la notebook que desarrolla el **Problema 3**, que abarca la implementación de modelos de redes neuronales para clasificación de imágenes.
   - [Link para descargar el dataset](https://drive.google.com/file/d/1Pqs5Y6dZr4R66Dby5hIUIjPZtBI28rmJ/view?usp=drive_link)

---

## Requisitos de Instalación

-  numpy 
-  matplotlib
-  seaborn
-  pandas
-  tensorflow
-  gdown
-  scikit-learn
