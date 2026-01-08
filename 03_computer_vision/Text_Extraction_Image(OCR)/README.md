# Text Extraction from Image (OCR)

## Problem
Extract readable text from an image using Optical Character Recognition (OCR).
This project converts image-based text into machine-readable text.

## Approach
This project uses classical computer vision techniques combined with
Tesseract OCR to improve text extraction accuracy.

The workflow includes:
1. Load input image
2. Convert image to grayscale
3. Apply image preprocessing (thresholding, noise removal)
4. Use Tesseract OCR to extract text from the processed image
5. Display and analyze the extracted text

## Techniques Used
- Image preprocessing with OpenCV
- Grayscale conversion
- Thresholding for text enhancement
- Optical Character Recognition (OCR) using Tesseract

## Output
- Extracted text printed as plain text
- Visual display of the processed image

## Files
- Project6_Text_Extract_from_Image.ipynb` — complete notebook implementation

## Tools & Libraries
- Python
- OpenCV (`cv2`)
- pytesseract
- NumPy
- Matplotlib
- PIL (Pillow)

## How to Run
1. Open the notebook:
   ```bash
   Project6_Text_Extract_from_Image.ipynb
