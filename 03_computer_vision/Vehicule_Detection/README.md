# Vehicle Detection using OpenCV (Haar Cascade)

## Problem
Detect vehicles in images and video streams using computer vision techniques.
The goal is to locate vehicles and draw bounding boxes around them.

## Approach
This project uses a classical computer vision approach based on Haar Cascade
classifiers provided by OpenCV.

The workflow includes:
1. Load image or video input
2. Convert frames to grayscale
3. Apply Haar Cascade classifier for vehicle detection
4. Draw bounding boxes around detected vehicles
5. Display detection results frame by frame

## Model / Method
- Haar Cascade Classifier (pretrained)
- Sliding window–based object detection

This approach does not require training a model from scratch and is suitable
for real-time detection in controlled environments.

## Input
- Static images
- Video files (processed frame by frame)

## Output
- Bounding boxes drawn around detected vehicles
- Visual display of detected vehicles in images/videos

## Key Concepts Used
- Object detection
- Haar Cascades
- Grayscale image processing
- Bounding box visualization
- Video frame processing

## Files
- Project_8_vehicule_detection.ipynb` — complete notebook implementation

## Tools & Libraries
- Python
- OpenCV (`cv2`)
- NumPy
- Matplotlib

## How to Run
1. Open the notebook:
   ```bash
   Project_8_vehicule_detection.ipynb
