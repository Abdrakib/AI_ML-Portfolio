# Plant Disease Classification (CNN)

## Problem
Identify and classify plant diseases from leaf images using deep learning.
This is a multi-class image classification problem where each class represents
a specific plant disease or healthy leaf.

## Dataset
Image dataset of plant leaves with multiple disease categories.
- Images are resized to a fixed input size
- Dataset is split into training and testing sets
- Each image belongs to one plant disease class

## Approach
This project uses a Convolutional Neural Network (CNN) trained from scratch.

Workflow:
1. Load and preprocess plant leaf images
2. Resize images to a fixed size
3. Normalize pixel values
4. Build a CNN architecture
5. Train the model on labeled images
6. Evaluate performance on test data
7. Visualize predictions

## Model
- Framework: TensorFlow / Keras
- Architecture: Custom CNN
  - Convolutional layers
  - MaxPooling layers
  - Fully connected (Dense) layers
- Output layer: Softmax (multi-class classification)
- Loss function: Categorical Cross-Entropy
- Optimizer: Adam

## Evaluation
- Metric used: Accuracy
- Training and validation accuracy monitored during training
- Model performance evaluated on unseen test images

## Results
The CNN learns visual patterns in plant leaves and achieves good
classification accuracy.
Some diseases with similar visual symptoms are more challenging to classify.

## Files
- project7_plant__Deaseases_prediction.ipynb` — complete notebook including:
  - Data preprocessing
  - CNN model training
  - Evaluation
  - Prediction visualization

## Tools & Libraries
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- OpenCV (for image handling)

## How to Run
1. Open the notebook:
   ```bash
   project7_plant__Deaseases_prediction.ipynb
