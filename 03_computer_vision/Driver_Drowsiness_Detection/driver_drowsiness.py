"""
Driver Drowsiness Detection
Detects drowsiness using eye aspect ratio and yawn detection
"""

import cv2
import numpy as np
import dlib

# initialize face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def eye_aspect_ratio(eye_points):
    """Calculate eye aspect ratio"""
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

# drowsiness detection
# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray)
#     
#     for face in faces:
#         landmarks = predictor(gray, face)
#         # extract eye points and calculate EAR
#         # if EAR < threshold: drowsy
#         # draw alert on frame
#     
#     cv2.imshow('Drowsiness Detection', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

print("Driver drowsiness detection script ready.")
