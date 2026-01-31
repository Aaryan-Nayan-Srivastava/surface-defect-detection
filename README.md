# 🛠️ Surface Defect Detection using CNN

## 📌 Project Overview

This project focuses on **automated surface defect detection in steel plates** using **Convolutional Neural Networks (CNNs)**.  
The objective is to classify steel surface images into one of six defect categories, helping improve quality inspection in manufacturing.

The current implementation uses a **custom CNN trained from scratch** on grayscale images.  
Future versions will explore **transfer learning with pretrained models** to improve performance and generalization on limited data.

---

## 🧠 Defect Classes

The model classifies images into the following categories:

1. Crazing  
2. Inclusion  
3. Patches  
4. Pitted Surface  
5. Rolled-in Scale  
6. Scratches  

---

## 📂 Dataset

- **Dataset**: NEU Surface Defect Database (NEU-DET)
- **Image Type**: Grayscale
- **Image Size**: 128 × 128
- **Number of Classes**: 6
- **Data Split**: Training and Validation sets

---

## 🏗️ Model Architecture

The model is a custom-built CNN implemented using TensorFlow / Keras.

**Architecture Overview:**
- Data Augmentation
- Image Rescaling (1/255)
- Convolutional Blocks:
  - Conv2D
  - Batch Normalization
  - MaxPooling
  - Dropout
- Global Average Pooling
- Fully Connected Dense Layer with L2 Regularization
- Softmax Output Layer (6 classes)

This design balances feature extraction, regularization, and computational efficiency.

---

## 📊 Model Performance

- **Validation Accuracy**: ~92%
- Strong and balanced performance across all defect classes
- Confusion matrix analysis shows high precision and recall with minor confusion between visually similar defect types

---

## 🖥️ Streamlit Web App

The trained model is deployed using **Streamlit** for interactive inference.

### Features:
- Upload steel surface images (JPG / PNG)
- Automatic preprocessing (grayscale, resize, normalization)
- Displays:
  - Predicted defect class
  - Confidence score
  - Raw class probabilities

---
Planned improvements for upcoming versions:

- Replace custom CNN with **pretrained models** (MobileNetV2, EfficientNet)
- Improve generalization on small datasets using transfer learning
- Add **Grad-CAM visualizations** for model interpretability
- Enhance Streamlit UI with probability charts and top-k predictions

---

## 📌 Key Learnings

- Built an end-to-end image classification pipeline
- Designed and trained a CNN from scratch
- Performed detailed error and confusion matrix analysis
- Identified and resolved preprocessing mismatches between training and inference
- Deployed the model as an interactive web application

---

⭐ If you find this project useful, feel free to star the repository!
