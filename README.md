<div align="center">

# 🔠 Handwritten Alphanumeric Character Recognition

### Computer Vision × Deep Learning × Real-Time Camera Inference

<p align="center">
  A computer vision system that recognizes handwritten
  <strong>digits (0–9)</strong> and
  <strong>uppercase letters (A–Z)</strong>
  using a custom PyTorch CNN and browser-based camera input.
</p>

<br>

<img src="https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/PyTorch-CNN-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white">
<img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5c3ee8?style=for-the-badge&logo=opencv&logoColor=white">
<img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white">
<img src="https://img.shields.io/badge/EMNIST-Dataset-8A2BE2?style=for-the-badge">

<br><br>

**36 Classes**

0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ

What is this?

This project is an end-to-end handwritten character recognition system
that combines computer vision, deep learning, and a browser camera into a
single pipeline.

A user writes a single digit or uppercase letter on paper and places it inside
the camera recognition area. The browser captures the character and sends the
image to a FastAPI backend.

OpenCV then extracts and normalizes the handwritten character before passing it
to a custom-trained convolutional neural network built with PyTorch.

The CNN predicts the character and returns a confidence score to the browser.





🚀 Quick Start

This section is for anyone who wants to clone the repository and run the
project locally.

1. Access the Repository

Clone the GitHub repository:
git clone https://github.com/YOUR-USERNAME/Computer-Vision-and-Readability.git

Enter the project directory:
cd Computer-Vision-and-Readability

Replace YOUR-USERNAME with the GitHub username that owns the repository.

```text
Handwritten Character
        ↓
Browser Camera
        ↓
Image Capture
        ↓
OpenCV Preprocessing
        ↓
Character Extraction
        ↓
28 × 28 Normalization
        ↓
PyTorch CNN
        ↓
36-Class Prediction
        ↓
Character + 


_________________________________________________________________________________________________________________________________




    Install all required Python packages:

    pip install -r requirements.txt

    This installs the libraries required for:

                PyTorch
                OpenCV
                NumPy
                FastAPI
                Uvicorn
                Image processing
                Model inference




_______________________________________________________________________________________________________






From the project root directory, run:
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000





___________________________________________________________________________________________________________-




Machine Learning Model

The recognition model is a custom Convolutional Neural Network built with
PyTorch.

Architecture
Input
28 × 28 × 1
     │
     ▼
Conv2D  1 → 32
     │
   ReLU
     │
 MaxPool
     │
     ▼
Conv2D  32 → 64
     │
   ReLU
     │
 MaxPool
     │
     ▼
Conv2D  64 → 128
     │
   ReLU
     │
 MaxPool
     │
     ▼
Flatten
     │
     ▼
Linear
1152 → 256
     │
   ReLU
     │
  Dropout
     │
     ▼
Linear
256 → 36
     │
     ▼
Prediction




_________________________________________________________________________________________________________________







Training Configuration


| Parameter            | Value           |
| -------------------- | --------------- |
| Dataset              | EMNIST Balanced |
| Selected Classes     | 36              |
| Input Size           | 28 × 28         |
| Batch Size           | 128             |
| Epochs               | 5               |
| Optimizer            | Adam            |
| Learning Rate        | 0.001           |
| Loss Function        | Cross Entropy   |
| Training Device      | CPU             |
| Validation Accuracy  | ~91.04%         |
| EMNIST Test Accuracy | ~92.50%         |
