"""
Human Pose Estimation
Detects keypoints in human body
"""

import cv2
import numpy as np

# using mediapipe or OpenPose
# import mediapipe as mp

# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()

def detect_pose(image):
    """Detect pose keypoints"""
    # results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # if results.pose_landmarks:
    #     for landmark in results.pose_landmarks.landmark:
    #         x = int(landmark.x * image.shape[1])
    #         y = int(landmark.y * image.shape[0])
    #         cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
    return image

print("Pose estimation script ready. Install mediapipe to use.")
