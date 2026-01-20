"""
Pedestrian Detection
Detects pedestrians in images/video using HOG features
"""

import cv2

# initialize HOG descriptor
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def detect_pedestrians(image):
    """Detect pedestrians in image"""
    # boxes, weights = hog.detectMultiScale(image, winStride=(8,8), padding=(32,32), scale=1.05)
    
    # for (x, y, w, h) in boxes:
    #     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return image

# video detection
# cap = cv2.VideoCapture('video.mp4')
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frame = detect_pedestrians(frame)
#     cv2.imshow('Pedestrian Detection', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

print("Pedestrian detection script ready.")
