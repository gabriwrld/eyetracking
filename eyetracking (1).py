import cv2
import dlib
import numpy as np
from math import hypot
import pyautogui
import time
import os
# faz reconhecimento da webcam
cap = cv2.VideoCapture(0)

# passo de verificação
if not cap.isOpened():
    print("Erro")
    exit()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:\\Users\\Gabriel\\Desktop\\python_codes\\shape_predictor.dat")
# Variáveis para controle de olhos fechados
eye_closed_start = None
keyboard_opened = False

def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    #hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 1)
    #ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 1)

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio

def open_virtual_keyboard():
    # Comando para abrir o teclado virtual no Windows
    os.system("osk")

def get_gaze_ratio(eye_points, facial_landmarks, gray, frame):
    # Define a região do olho com base nos pontos fornecidos
    eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                           (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                           (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                           (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                           (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                           (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
        
    # Visualiza a região do olho no frame
    cv2.polylines(frame, [eye_region], True, (0, 0, 255), 2)

    # Cria uma máscara para o olho
    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [eye_region], 255)

    eye = cv2.bitwise_and(gray, gray, mask=mask)

    # Define os limites do olho
    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY_INV)

    # Divide o olho em lado esquerdo e direito
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    # Calcula o gaze ratio
    gaze_ratio = left_side_white / (right_side_white + 1e-6)  # Adiciona um pequeno valor para evitar divisão por zero
    return gaze_ratio


font = cv2.FONT_HERSHEY_SIMPLEX

while True:

    # faz a leitura da captura de video da webcam
    ret, frame = cap.read()

    #passo de verificação
    if not ret:
        print("Erro na leitura")
        break

    frame = cv2.flip(frame, 1)

    # converte o frame da camera em escala de cinzas
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        #x, y = face.left(), face.top()
        #x1, y1 = face.right(), face.bottom()
        #cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        landmarks = predictor(gray, face)

        #detecção de piscadas
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio)/2 

        if blinking_ratio > 5.3:
            cv2.putText(frame, "PISCANDO", (50, 400), font, 1, (255, 0, 0), 3)
            if eye_closed_start is None:
                eye_closed_start = time.time()  # Marca o início do olho fechado
        else:
            eye_closed_start = None  # Reseta se o olho não está mais fechado
            keyboard_opened = False  # Reseta o estado do teclado

        # Verifica se o olho está fechado por mais de 1.5 segundos
        if eye_closed_start is not None:
            duration = time.time() - eye_closed_start
            if duration > 1.0:
                cv2.putText(frame, "OLHO FECHADO", (50, 450), font, 1, (0, 0, 255), 3)

            # Abre o teclado virtual após 2 segundos
            if duration > 1.5 and not keyboard_opened:
                keyboard_opened = True
                open_virtual_keyboard()

        #detecção da íris
        #gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
        #gaze_ratio_right_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
        gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks, gray, frame)
        gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks, gray, frame)
        gaze_ratio = (gaze_ratio_left_eye + gaze_ratio_right_eye) / 2

        #COMEÇO ALTERAÇÃO

        # Detecção de piscadas para o olho direito
        if right_eye_ratio > 5.0:
            cv2.putText(frame, "OLHO DIREITO FECHADO", (50, 300), font, 1, (0, 255, 255), 3)
            if eye_closed_start is None:
                eye_closed_start = time.time()  # Marca o início do olho fechado
        else:
            eye_closed_start = None  # Reseta o estado se o olho não está mais fechado

        if gaze_ratio_left_eye > 1.37:
            cv2.putText(frame, "ESQUERDA", (50, 100), font, 1, (0, 0, 255), 3)
            pyautogui.moveRel(-30, 0, duration = 0.25)
        elif gaze_ratio_left_eye < 0.51:
            cv2.putText(frame, "DIREITA", (50, 100), font, 1, (0, 0, 255), 3)
            pyautogui.moveRel(30, 0, duration = 0.25)
        elif gaze_ratio_left_eye < 0.63 and gaze_ratio_left_eye > 0.53:
            cv2.putText(frame, "BAIXO", (50, 100), font, 1, (0, 0, 255), 3)
            pyautogui.moveRel(0, 30, duration = 0.25)
        elif gaze_ratio_left_eye > 0.66 and gaze_ratio_left_eye < 1.2:
            cv2.putText(frame, "CENTRO", (50, 100), font, 1, (0, 0, 255), 3)
        elif gaze_ratio_left_eye > 1.22 and gaze_ratio_left_eye <1.34:
            cv2.putText(frame, "CIMA", (50, 100), font, 1, (0, 0, 255), 3)
            pyautogui.moveRel(0, -30, duration = 0.25)
  

    # abre uma janela para a webcam no tamanho 640x360
    cv2.imshow("Camera", frame)
    frame = cv2.resize(frame, (640, 360)) 
    cv2.waitKey(1)
    if cv2.getWindowProperty("Camera", cv2.WND_PROP_VISIBLE) < 1:
        break

# encerra 
cv2.destroyAllWindows()
cap.release()