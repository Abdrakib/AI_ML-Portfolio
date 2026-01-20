"""
Face Mask Detection
Detects if person is wearing a mask or not using real-time video
"""

import cv2
import numpy as np
from tensorflow import keras

# load pre-trained model
# model = keras.models.load_model('mask_detection_model.h5')

# face detection using Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_mask(frame):
    """Detect mask on faces in the frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        # resize to model input size
        face_resized = cv2.resize(face_roi, (150, 150))
        face_resized = face_resized / 255.0
        face_resized = np.expand_dims(face_resized, axis=0)
        
        # predict
        # prediction = model.predict(face_resized)
        # if prediction[0][0] > 0.5:
        #     label = "Mask"
        #     color = (0, 255, 0)
        # else:
        #     label = "No Mask"
        #     color = (0, 0, 255)
        
        # draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    return frame

# real-time detection
# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frame = detect_mask(frame)
#     cv2.imshow('Face Mask Detection', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

print("Face mask detection script ready. Load your trained model and dataset.")
