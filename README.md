# 100-Class Sports Classification using Custom CNN

## 📌 Overview

This project implements a 100-class sports classification model using a custom Convolutional Neural Network (CNN). The dataset contains 13,492 training images, 500 validation images, and 500 test images, each resized to 224x224 pixels. The goal is to classify images into one of 100 different sports categories.

The model has been designed from scratch without using any pre-trained architectures. The approach focuses on building a deep learning model that can generalize well across a large variety of sports categories.

## 🛠️ Setup and Installation

### 1️⃣ Install Dependencies

Ensure you have Python 3.10 and required libraries installed:

pip install tensorflow numpy matplotlib seaborn scikit-learn

### 2️⃣ Clone the Repository

git clone https://github.com/Sakib179/100-class-sports-classification.git
cd 100-class-sports-classification

## 🔍 Project Methodology

### 📂 Dataset Preparation

The dataset is structured into training, validation, and test sets to ensure robust model evaluation. Images are resized to 224x224 pixels to maintain consistency in input size. Data preprocessing techniques such as normalization and augmentation were applied to improve generalization.

### 🏗️ Model Design

A custom CNN architecture was developed to effectively capture spatial patterns within the images. The network consists of multiple convolutional layers, max-pooling layers, and dropout layers to enhance feature extraction and reduce overfitting. The final dense layers map the learned features to 100 output classes using the softmax activation function.

### 🎯 Training and Evaluation

The model was trained using categorical cross-entropy loss and the Adam optimizer. Performance was measured using validation accuracy and loss trends. A test dataset was used to evaluate the model’s final accuracy and robustness.

### 📊 Results and Performance

The model successfully classifies 100 different sports categories.

Training and validation accuracy trends were analyzed to fine-tune hyperparameters.

Performance metrics such as precision, recall, and F1-score were calculated for deeper insights.

## 🚀 Future Improvements

To further enhance the model's performance and robustness, the following improvements are proposed:

    1. Transfer Learning – Utilizing pre-trained models like ResNet, EfficientNet, or DenseNet to improve feature extraction.

    2. Data Augmentation – Expanding the dataset with synthetic variations to improve generalization and reduce overfitting.

    3. Hyperparameter Tuning – Experimenting with different learning rates, batch sizes, and activation functions to optimize model performance.

    4. Additional Regularization Techniques – Implementing techniques such as dropout, L2 regularization, and batch normalization more effectively.

    5. Better Model Interpretability – Using Grad-CAM and SHAP techniques to visualize which parts of the image contribute most to classification.

    6. Deployment & Real-Time Prediction – Converting the trained model into a lightweight version using TensorFlow Lite or ONNX for real-world applications.
