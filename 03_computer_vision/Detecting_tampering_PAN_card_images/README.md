# PAN Card Tampering Detection (Image Processing)

## Problem
Detect whether a PAN card image has been tampered with by comparing
an original image and a modified (tampered) image.

The goal is to identify visual differences and highlight the regions
where tampering has occurred.

## Approach
This project uses classical image processing techniques to compare two
images and detect structural differences.

The workflow includes:
1. Load original and tampered PAN card images
2. Convert images to grayscale
3. Resize images to the same dimensions
4. Compute the Structural Similarity Index (SSIM)
5. Identify differences between images
6. Detect contours around tampered regions
7. Draw bounding boxes around detected tampering areas

## Techniques Used
- Image resizing and normalization
- Grayscale image processing
- Structural Similarity Index (SSIM)
- Thresholding
- Contour detection
- Bounding box visualization

## Output
- Difference image highlighting tampered regions
- Bounding boxes drawn around detected tampered areas
- Visual comparison between original and tampered PAN cards

## Files
- Pan_Card_Tempering_Project.ipynb` — complete notebook implementation

## Tools & Libraries
- Python
- OpenCV (`cv2`)
- NumPy
- scikit-image (SSIM)
- Matplotlib
- imutils

## How to Run
1. Open the notebook:
   ```bash
   Pan_Card_Tempering_Project.ipynb
