"""
Gender and Age Detection
Predicts gender and age from face images
"""

import cv2
import numpy as np

# load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# model would be loaded here
# gender_model = load_gender_model()
# age_model = load_age_model()

def detect_gender_age(image):
    """Detect gender and age from image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        face_roi = image[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (64, 64))
        face_resized = face_resized / 255.0
        face_resized = np.expand_dims(face_resized, axis=0)
        
        # predict gender and age
        # gender_pred = gender_model.predict(face_resized)
        # age_pred = age_model.predict(face_resized)
        
        # gender = "Male" if gender_pred[0][0] > 0.5 else "Female"
        # age = int(age_pred[0][0])
        
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # cv2.putText(image, f"{gender}, {age}", (x, y-10), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return image

print("Gender and age detection script ready.")
