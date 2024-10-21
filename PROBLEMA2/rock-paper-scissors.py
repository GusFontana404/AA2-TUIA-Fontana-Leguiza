import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import argparse
import os

# Crear un objeto ArgumentParser
parser = argparse.ArgumentParser(description="Script para probar modelo de detección de gestos.")

# Añadir un argumento para la ruta
parser.add_argument('--model_path',  default=os.getcwd(), help='Ruta para cargar el modelo entrenado.')

# Parsear los argumentos
args = parser.parse_args()

# Usar las rutas proporcionadas
model_path = args.model_path

# Cargar el modelo entrenado
model = tf.keras.models.load_model(model_path)

# Iniciar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

# # Inicializar MediaPipe para la detección de manos 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Parámetros para la detección de manos
hands = mp_hands.Hands(static_image_mode = False,
                      max_num_hands = 1,
                      min_detection_confidence = 0.8,
                      min_tracking_confidence = 0.8)

gesture_map = {0: "Piedra", 1: "Papel", 2: "Tijeras"}

while True:
    # Leer el frame de la cámara
    ret, frame = cap.read()
    
    # Espejar (voltear horizontalmente) el frame
    frame = cv2.flip(frame, 1)

    if not ret:
        print("Error: No se pudo capturar el frame.")
        break

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la imagen
    results = hands.process(frameRGB)

    # Dibujar puntos de referencias si encuentra una mano
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])

            # Convertir landmarks a numpy array y predecir el gesto
            landmarks_np = np.array(landmarks).flatten().reshape(1, 21, 3)
            prediction = model.predict(landmarks_np)
            gesture = np.argmax(prediction, axis=1)[0]

            # Mostrar el gesto en la pantalla
            cv2.putText(frame, gesture_map[gesture], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Mostrar el frame en una ventana
    cv2.imshow('Camara', frame)

    # Salir del loop si se presiona la tecla 'q'
    if cv2.waitKey(1) == ord("q"):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()