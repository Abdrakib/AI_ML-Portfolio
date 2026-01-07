# Face Swap Application (OpenCV + dlib)

## Problem
Swap the face from one image onto another image while keeping the result natural-looking.
This project uses facial landmarks, triangulation, warping, and seamless blending.

## What This Project Does
1. Loads 2 images (source face and target face)
2. Detects faces using dlib’s frontal face detector
3. Extracts 68 facial landmarks using a pretrained landmark model
4. Creates a facial mask using the convex hull of landmarks
5. Applies Delaunay triangulation on landmark points
6. Warps each triangle from the source face to the target face (affine transform)
7. Reconstructs the full swapped face
8. Blends the swapped face into the target image using `cv2.seamlessClone`

## Key Techniques
- Face detection (dlib)
- 68 facial landmark detection (`shape_predictor_68_face_landmarks.dat`)
- Convex hull face masking (OpenCV)
- Delaunay triangulation (`cv2.Subdiv2D`)
- Triangle-by-triangle warping (affine transform)
- Seamless blending (`cv2.seamlessClone`)

## Files
- `project9_FaceSwapApplication.ipynb` — complete notebook implementation

## Tools & Libraries
- Python
- OpenCV (`cv2`)
- dlib
- NumPy
- requests
- PIL (Pillow)

## How to Run
1. Open the notebook:
   - `project9_FaceSwapApplication.ipynb`
2. Make sure dependencies are installed:
   ```bash
   pip install opencv-python numpy pillow requests
