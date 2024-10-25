# Clasificador de Gestos: Piedra, Papel o Tijeras
Este proyecto implementa un sistema de clasificación de gestos utilizando la biblioteca MediaPipe para la detección de manos y un modelo de red neuronal densa para la clasificación de gestos de "piedra", "papel" o "tijeras".

## **Contenido del Directorio**

**El directorio contiene tres scripts principales:**

1-  record-dataset.py: Este script genera un dataset mediante el uso de la cámara. Permite guardar las coordenadas de los gestos requeridos en carpetas etiquetadas según su clase. Los archivos de coordenadas se guardan con la extensión .npy, almacenando las posiciones de los puntos de referencia de la mano definidos por la biblioteca MediaPipe.

2-  train-gesture-classifier.py: Este script entrena un modelo simple de capas densas totalmente conectado para clasificar los gestos. Recibe dos argumentos:
-  --dataset_path: Ruta donde se encuentra el dataset de entrenamiento.
-  --model_output_path: Ruta donde se guardará el modelo entrenado.

3-  rock-paper-scissors.py: Este script utiliza la cámara en streaming para ejecutar el modelo de clasificación entrenado y clasificar los gestos detectados. El argumento necesario es la ruta del modelo entrenado:
-  --model_path: Ruta para cargar el modelo entrenado.

## **Requisitos**
Para ejecutar los scripts, necesitarás instalar las siguientes bibliotecas:
- opencv-python
- mediapipe
- numpy
- tensorflow
