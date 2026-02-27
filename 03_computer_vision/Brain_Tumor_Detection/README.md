# Brain Tumor Detection

This project detects brain tumors from MRI images using Convolutional Neural Networks (CNN).

## Overview
The model classifies brain MRI images into two categories:
- Tumor present (yes)
- No tumor (no)

## Model Architecture
- Convolutional layers for feature extraction
- Max pooling for dimensionality reduction
- Fully connected layers for classification
- Binary classification output

## Usage
1. Prepare your dataset with 'yes' and 'no' folders
2. Update the data_dir path in the notebook
3. Run all cells to train the model

## Requirements
- TensorFlow/Keras
- OpenCV
- NumPy
- Matplotlib
- scikit-learn
