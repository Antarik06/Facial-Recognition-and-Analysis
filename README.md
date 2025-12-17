🔍 Advanced Face Recognition & Analysis System (Webcam-Based)

📌 Overview

This project is a modern, real-time face recognition and facial analysis system built using deep learning and computer vision.
The system has been fully redesigned and upgraded to use industry-grade models and now runs seamlessly on a laptop / USB webcam.

The project demonstrates face detection, face recognition, liveness verification, emotion detection, and real-time performance monitoring — all in a modular and scalable architecture.

🚀 Key Features

🎯 Face Detection using MTCNN (Deep Learning)

🧠 Face Recognition using FaceNet embeddings + SVM classifier

👁️ Blink Detection (Liveness / Anti-Spoofing) using MediaPipe

🙂 Emotion Detection (Neutral, Smiling, Surprised)

⚡ Performance Overlay (Real-time FPS)

📷 Webcam-based (Cross-platform)


🛠️ Tech Stack

Python 3

OpenCV

MTCNN

FaceNet (facenet-pytorch)

MediaPipe

Scikit-learn

NumPy

Torch

⚙️ Installation & Setup

1️⃣ Clone the repository

2️⃣ Install dependencies

pip install opencv-python mediapipe mtcnn facenet-pytorch torch torchvision scikit-learn numpy

▶️ How to Run

🔹 Run the complete system demo
python main_demo.py

🔹 What you’ll see:

Face bounding box (MTCNN)

Blink verification status

Emotion label

Real-time FPS counter

Press ESC to exit.

🧠 How It Works (Pipeline)

Webcam Frame
   ↓
MTCNN (Face Detection)
   ↓
MediaPipe (Blink + Emotion)
   ↓
FaceNet (Face Embeddings)
   ↓
SVM Classifier (Identity Prediction)
   ↓
Performance Overlay (FPS)

🎯 Why This Project Matters

This project goes beyond basic face detection and demonstrates:

Practical use of deep learning models

Understanding of liveness detection

Real-time system optimization

Clean, modular software design

It reflects real-world face recognition systems, not just academic demos.