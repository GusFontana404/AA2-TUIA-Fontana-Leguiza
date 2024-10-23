# Trabajo Práctico I - Aprendizaje Autómatico II

## Integrantes:
-  Fontana, Gustavo
-  Leguiza, Claudia

## **Redes Densas y Convolucionales**

Este repositorio contiene el desarrollo de tres problemáticas que tienen como objetivo final su resolución mediante el uso de arquitecturas de redes neuronales densas y convolucionales.

## **Objetivos**
El trabajo tiene como objetivos principales:

-  Construir un modelo de regresión utilizando redes neuronales para predecir el índice de rendimiento académico de los estudiantes basado en las características proporcionadas.

-  Implementar un sistema de clasificación de gestos de "piedra", "papel" o "tijeras" utilizando MediaPipe para la detección de las manos y una red neuronal densa para realizar la clasificación.

-  Construir un modelo de clasificación utilizando redes neuronales convolucionales (CNN) para clasificar imágenes en una de las seis categorías predefinidas.

## **Contenido del Repositorio**

1-  **Carpeta PROBLEMA1**: Contiene dentro una notebook con el desarrollo del problema 1, incluyendo el análisis exploratorio, la implementación del modelo y la evaluación de métricas.
Además contien el dataset, un archio csv.

2-  **Carpeta PROBLEMA2**: Contiene tres scripts para el desarrollo del problema 2 que permiten grabar un dataset, enternar un modelo y probarlo.
-  **record-dataset.py**
-  **rock-paper-scissors.py**
-  **train-gesture-classifier.py**

3-  **Carpeta PROBLEMA3**: Esta carpeta contiene la notebook que desarroll el problema 3, ademas de las imágenes utilizadas separadas en datasets de entrenamiento, testeo y predcción.

## **Requisitos para desplegar el contenido del repositorio**:
-  gdown==5.2.0
-  matplotlib==3.9.0
-  numpy==1.26.4
-  pandas==2.2.2
-  seaborn==0.13.2
-  scikit-learn==1.5.0
-  tensorflow==2.16.1
