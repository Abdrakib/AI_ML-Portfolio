# Traffic Sign Classification (CNN)

## Problem
Classify traffic sign images into their correct categories using a
Convolutional Neural Network (CNN).
This is a multi-class image classification problem commonly used in
autonomous driving and intelligent transportation systems.

## Dataset
Image dataset containing multiple classes of traffic signs.
- Images are resized to a fixed input size
- Dataset is split into training and testing sets
- Each image belongs to one traffic sign class

## Approach
This project uses a deep learning approach with a CNN trained from scratch.

Workflow:
1. Load and preprocess traffic sign images
2. Resize images to a fixed size
3. Normalize pixel values
4. Build a CNN model
5. Train the model on labeled traffic sign images
6. Evaluate model performance on test data
7. Visualize predictions and results

## Model
- Framework: TensorFlow / Keras
- Architecture: Custom Convolutional Neural Network
  - Convolutional layers
  - MaxPooling layers
  - Fully connected (Dense) layers
- Output layer: Softmax (multi-class classification)
- Loss function: Categorical Cross-Entropy
- Optimizer: Adam

## Evaluation
- Metric used: Accuracy
- Model performance evaluated on test data
- Prediction results visualized for sample images

## Results
The CNN successfully learns traffic sign features and achieves good
classification accuracy on the test dataset.
Some visually similar signs remain challenging.

## Files
- Traffic_Sign_classification.ipynb` — complete notebook including:
  - Data preprocessing
  - Model training
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
   Traffic_Sign_classification.ipynb
