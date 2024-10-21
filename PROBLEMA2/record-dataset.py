import cv2
import mediapipe as mp
import numpy as np
import os

# Iniciar la captura de video desde la cámara
cap = cv2.VideoCapture(0)

# Inicializar MediaPipe para la detección de manos 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Parámetros para la detección de manos
hands = mp_hands.Hands(static_image_mode = False,
                      max_num_hands = 1,
                      min_detection_confidence = 0.8,
                      min_tracking_confidence = 0.8)

# Inicializar contadores y ruta de almacenamiento
output_folder = os.path.join(os.getcwd(), 'dataset_rock-paper-scissors')
os.makedirs(output_folder, exist_ok=True)

gestures = ['rock', 'paper', 'scissors']
for gesture in gestures:
    gesture_dir = os.path.join(output_folder, gesture)
    os.makedirs(gesture_dir, exist_ok=True)

gesture_counters = {
    'rock': 0,
    'paper': 0,
    'scissors': 0
}

while True:
    # Leer el frame de la cámara
    ret, frame = cap.read()

    # Espejar (voltear horizontalmente) el frame
    frame = cv2.flip(frame, 1)

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la imagen
    results = hands.process(frameRGB)

    # Dibujar puntos de referencias si encuntra una mano
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    # Mostrar el frame en una ventana
    cv2.imshow('Camara', frame)

    # Leer la tecla presionada
    key = cv2.waitKey(1) & 0xFF

    # Capturar gestos y clasificarlos
    gesture = None
    if results.multi_hand_landmarks:
        if key == ord('r') or key == ord('R'):
            gesture = "rock"
        
        if key == ord('p') or key == ord('P'):
            gesture = "paper"
        
        if key == ord('s') or key == ord('S'):
            gesture = "scissors"
        
        if gesture:
            gesture_counters[gesture] += 1
            
            for hand_landmarks in results.multi_hand_landmarks:
                # Extraer coordenadas de los landmarks
                landmark_coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

            # Guardar las coordenadas en un archivo .npy
            coords_filename = os.path.join(f'{output_folder}/{gesture}',  f'{gesture}_{gesture_counters[gesture]}.npy')
            np.save(coords_filename, landmark_coords)

            print(f"Se ha capturado un gesto de {gesture}")

    # Salir del loop si se presiona la tecla 'q'
    if key == ord("q"):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()

print("\nFINALIZANDO CAPTURA DE GESTOS...")
print(f"DATASET GENERADO EN: {output_folder}\n")