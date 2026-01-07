# Dog Breed Prediction (Image Classification)

## Problem
Predict the breed of a dog from an input image using deep learning.
This is a multi-class image classification problem where each class
represents a different dog breed.

## Dataset
A labeled dataset of dog images organized by breed.
- Images are divided into training and validation sets
- Each image belongs to one dog breed class

## Approach
This project follows a progressive deep learning workflow:

1. Load and preprocess image data
2. Resize images to a fixed input size
3. Normalize pixel values
4. Train a Convolutional Neural Network (CNN) **from scratch** as a baseline
5. Apply **transfer learning using MobileNet** to improve performance
6. Compare baseline and transfer learning results

## Models
### 1. CNN from Scratch
- Custom convolutional layers
- MaxPooling layers
- Fully connected (Dense) layers
- Used as a baseline model

### 2. Transfer Learning (MobileNet)
- Pretrained MobileNet architecture
- Feature extraction using pretrained weights
- Custom classification head for dog breed prediction
- Fine-tuned for better accuracy and generalization

## Training Details
- Loss function: Categorical Cross-Entropy
- Optimizer: Adam
- Output layer: Softmax (multi-class classification)

## Evaluation
- Metric used: Accuracy
- Training and validation accuracy monitored during training
- MobileNet outperforms the CNN trained from scratch

## Results
- CNN from scratch provides a strong learning baseline
- Transfer learning with MobileNet significantly improves performance
- Some visually similar breeds remain challenging

## Files
- `Dog_Breed_prediction_.ipynb` — complete notebook including:
  - Data preprocessing
  - CNN from-scratch model
  - MobileNet transfer learning model
  - Training and evaluation

## Tools & Libraries
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- (Optional) OpenCV

## How to Run
1. Open the notebook:
   ```bash
   Dog_Breed_prediction_.ipynb
