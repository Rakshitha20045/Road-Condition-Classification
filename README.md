# Road-Condition-Classification
# Overview
This project leverages Convolutional Neural Networks (CNNs) for classifying road conditions based on images. The goal is to identify the state of roads to aid in effective maintenance and improve transportation infrastructure. The model classifies roads into different categories such as "Very Poor," "Poor," "Satisfactory," and "Good," based on visual inputs.The model is trained using a dataset of road images, and the output helps in determining the severity of road damage, enabling authorities to prioritize repairs.

# Project Description
The project involves the following key steps:

Data Preprocessing: Images are resized, normalized, and augmented to improve model performance.
Model Architecture: A CNN model is designed to process road images, extract features, and classify road conditions.
Training: The model is trained on a labeled dataset using backpropagation and optimization techniques.
Evaluation: The model is tested on unseen images to evaluate its classification accuracy and performance metrics such as confusion matrix.

# Dataset

The dataset used for road condition classification contains 1,272 images of roads labeled according to their condition:
Very Poor: Roads with severe damage like potholes and large cracks.
Poor: Roads with moderate damage or degradation.
Satisfactory: Roads with minimal damage or wear.
Good: Roads in perfect condition

# Total images: 1,272
# Training images: 1,016 (80% of the dataset)
# Testing images: 256 (20% of the dataset)
# Source : https://www.kaggle.com/datasets/prudhvignv/road-damage-classification-and-assessment/data

# Setup and Installation

# Prerequisites:
  Python 3.7+
  TensorFlow/Keras 2.x
  NumPy
  Matplotlib
  scikit-learn
  Seaborn

# Installation Steps:
1.Clone this repository
2.Install the required dependencies
pip install tensorflow==2.10.0 numpy==1.23.5 matplotlib==3.6.3 scikit-learn==1.1.3 seaborn==0.11.2 Pillow==9.2.0 h5py==3.7.0
3.Download the trained models: 

4.Once the model is trained, evaluate it on the test dataset based pn
    Accuracy
    Confusion Matrix
    Classification Report

# Results:
# 1.CNN Architecture
   Training Accuracy: 0.9724
   Test Accuracy: 0.6172
   Architecture : The CNN model consists of 3 convolutional layers, followed by 3 dense layers 
   with 128, 64, and 32 neurons, and an output layer of 7 neurons using the softmax activation 
   function for classification.

# Results Visualization 

![Image](https://github.com/user-attachments/assets/b665b729-b09e-4cf8-b6d0-8054d38bf92d)
![Image](https://github.com/user-attachments/assets/4044d4fa-d21f-45ee-a95f-b244fca6bdcd)
![Image](https://github.com/user-attachments/assets/6af2ef22-1c27-446e-a5ab-84787a756029)
![Image](https://github.com/user-attachments/assets/942ee755-880f-4976-b5b9-3f3caaba563f)
![Image](https://github.com/user-attachments/assets/38e35e2e-b7d7-4175-9e06-5f27a7f7f1f2)



