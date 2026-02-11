# Spatial-Coordinate-Regression-CNN
This project implements a Deep Learning solution to predict the (x, y)coordinates of a single active pixel (value 255) within a (50*50)grayscale grid. This is a Supervised Regression task focused on spatial localization.

🚀 Overview
The model takes a (50*50) image as input and outputs two continuous values representing the horizontal (x) and vertical (y) positions of the white pixel.

Key Features

1)Approach: Convolutional Neural Network (CNN) for spatial feature extraction.
2)Data: 100% synthetically generated to cover the entire search space.
3)Precision: Achieves sub-pixel accuracy through normalized regression.
4)Standards: PEP8 compliant code with detailed inline documentation.

📊 Methodology & Rationale

1.Dataset Choice
Instead of a random sample, I generated a complete synthetic dataset of 2,500 images.
Coverage: Every possible pixel location (0-49, 0-49) is represented once.
Normalization: Target coordinates are scaled to a [0, 1] range. This improves gradient stability and allows the model to treat the grid as a continuous coordinate system rather than discrete categories.

2. Why Regression over Classification?
Efficiency: Classification would require 2,500 output classes. Regression requires only 2 output nodes.
Spatial Awareness: Regression uses Mean Squared Error (MSE), which penalizes the model based on the distance from the true pixel. This helps the model learn the concept of "proximity."

3. Model Architecture
I utilized a CNN architecture because it is the industry standard for image-based tasks.
Conv2D Layers: Used to detect high-intensity contrast points.
Global Average Pooling: Used to reduce parameter count and prevent overfitting.
Sigmoid Activation: Used in the final layer to bound the coordinate predictions between 0 and 1.

🛠️ Installation & Usage
Prerequisites:
Python 3.8+
Google Colab (Recommended) or Jupyter Notebook

Dependencies:
Install the required libraries using pip:
pip install numpy matplotlib tensorflow scikit-learn

📈 Results
The notebook includes:
Training Logs: Real-time MSE loss and MAE tracking.

Loss Curves: Visualization of model convergence over 20 epochs.

Prediction Plots: Comparison of Ground Truth (Green Circle) vs. Model Prediction (Red X) on unseen test data.
Training Logs: Real-time MSE loss and MAE tracking.
Loss Curves: Visualization of model convergence over 20 epochs.
Prediction Plots: Comparison of Ground Truth (Green Circle) vs. Model Prediction (Red X) on unseen test data.
