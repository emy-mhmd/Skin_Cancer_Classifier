# Skin Cancer Classification using Transfer Learning and Handcrafted CNN  

This repository contains the implementation of a skin cancer binary classification project using transfer learning models and a handcrafted CNN. The models are designed to classify images as either benign or malignant with high accuracy. Additionally, the project includes a user-friendly web application that allows users to upload skin lesion images and select a model for predictions, providing a confidence percentage for each result.  

### Key Features:  
- **Pre-Trained Models**: Implementations of ResNet50, EfficientNetB5, MobileNetV2, and VGG16, trained on the ISIC dataset for binary classification.  
- **Handcrafted CNN**: A custom-designed convolutional neural network inspired by VGG16 architecture, achieving an accuracy of 87%.  
- **Ensemble Approach**: Combines predictions from all models to improve overall accuracy and robustness.  
- **Web Interface**: A simple and interactive web application for model selection and image classification, making AI-assisted diagnostics accessible.  
- **Visualization Tools**: Confusion matrices, ROC curves, and loss/accuracy graphs to evaluate model performance.  

### Getting Started:  
The repository includes all necessary scripts for model training, evaluation, and deployment.  

## Dataset: 
[skin-cancer-isic-images](https://www.kaggle.com/datasets/rm1000/skin-cancer-isic-images/data)

## Pretrained Models Weights
[Find weights here](https://drive.google.com/drive/folders/1QVDyVlESDp_Q1T7Swl4abTkB5AEWl5PM?usp=sharing)

## Install Requirements

```
pip install flask flask-cors torch torchvision

```

## Run server

```
python server.py

```

## Run Frontend

Open index.html in any browser

## Skin Cancer Detector Demo
https://github.com/user-attachments/assets/0d9f6a49-1e38-49ce-8279-5cbbd908cec6

