import numpy as np
from pathlib import Path
import os
import argparse
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, Input, Dropout

# Crear un objeto ArgumentParser
parser = argparse.ArgumentParser(description="Script para entrenar un modelo de clasificación de gestos.")

# Añadir un argumento para la ruta
parser.add_argument('--dataset_path', type=str, default=os.getcwd(), help='Ruta donde se encuentra el dataset de entrenamiento.')
parser.add_argument('--model_output_path', type=str, default=os.getcwd(), help='Ruta donde se guardará el modelo entrenado.')

# Parsear los argumentos
args = parser.parse_args()

# Usar las rutas proporcionadas
dataset_path = args.dataset_path
model_output_path = args.model_output_path

# Asegurarse de que el directorio de salida existe
os.makedirs(model_output_path, exist_ok=True)

# Mapeo de gestos a etiquetas
gesture_map = {"rock": 0, "paper": 1, "scissors": 2}

# Cargar archivos .npy y asignar etiquetas
landmarks, labels = [], []
    
# Recorrer las subcarpetas 
for gesture_name, label in gesture_map.items():
    gesture_dir = os.path.join(dataset_path, gesture_name)
        
    # Recorrer los archivos en la carpeta correspondiente a cada clase
    for filename in os.listdir(gesture_dir):
        if filename.endswith('.npy'):
            filepath = os.path.join(gesture_dir, filename)
            landmarks.append(np.load(filepath))  
            # Asignar el label basado en la carpeta
            labels.append(label)  

# Convertir las listas a arrays de numpy
landmarks_array, labels_array= np.array(landmarks), np.array(labels)

# Tamaño del lote y número de clases
batch_size, num_clases = 32, 3

# Dividir el dataset en entrenamiento y validación (80% - 20%)
train_size = int(0.8 * len(labels_array))
train_landmarks = landmarks_array[:train_size]
train_labels = labels_array[:train_size]
val_landmarks = landmarks_array[train_size:]
val_labels = labels_array[train_size:]

dense_model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(21, 3)), # 21 puntos clave
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(.5),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(num_clases, activation="softmax"),
    ]
)

dense_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

# Número de épocas de entrenamiento
dense_epoch = 400

print("\n\nINICIANDO EL ENTRENAMIENTO...\n")

# Entrena el modelo
dense_history = dense_model.fit(
    train_landmarks,
    train_labels,
    validation_data=(val_landmarks, val_labels),
    epochs=dense_epoch,
    batch_size=batch_size,
    shuffle=True,
)

print("\n\nENTRENAMIENTO FINALIZADO\n")

# Guardar el modelo completo
dense_model.save(f'{model_output_path}/gesture_classifier_model.keras')
print(f"Modelo guardado en {model_output_path}\n\n")