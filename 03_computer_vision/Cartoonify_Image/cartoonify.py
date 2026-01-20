"""
Cartoonify Image
Converts images to cartoon style
"""

import cv2
import numpy as np

def cartoonify(image):
    """Convert image to cartoon style"""
    # bilateral filter for smoothing
    filtered = cv2.bilateralFilter(image, 9, 75, 75)
    
    # convert to grayscale
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    
    # edge detection
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY, 9, 9)
    
    # color quantization
    data = image.reshape((-1, 3))
    data = np.float32(data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, 8, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()].reshape(image.shape)
    
    # combine edges and quantized colors
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(quantized, edges_colored)
    
    return cartoon

# test
# img = cv2.imread('input.jpg')
# cartoon = cartoonify(img)
# cv2.imwrite('cartoon.jpg', cartoon)

print("Image cartoonification script ready.")
