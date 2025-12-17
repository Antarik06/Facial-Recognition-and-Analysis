## 🔍 Advanced Face Recognition & Analysis System (Webcam-Based)

### 📌 Overview

This project is a real-time face recognition and facial analysis system built using modern computer vision and deep learning techniques.  
It has been redesigned from the ground up to use **industry-grade models** and runs smoothly on a standard laptop or USB webcam.

The system performs **face detection, identity recognition, liveness verification, emotion detection, and real-time performance tracking**, all structured in a clean and modular pipeline suitable for real-world use cases.

---

## 🚀 Key Features

- **Face Detection** using **MTCNN** (deep learning–based, robust to lighting & angles)
- **Face Recognition** using **FaceNet embeddings** with an **SVM classifier**
- **Blink Detection** for **liveness / anti-spoofing** using **MediaPipe**
- **Basic Emotion Detection** (Neutral, Smiling, Surprised)
- **Real-time FPS overlay** for performance monitoring
- **Webcam-based & cross-platform**

---

## 🛠️ Tech Stack

- Python 3  
- OpenCV  
- MTCNN  
- FaceNet (facenet-pytorch)  
- MediaPipe  
- Scikit-learn  
- NumPy  
- PyTorch  

---

## 🧠 System Pipeline

```text
Webcam Frame
     ↓
MTCNN – Face Detection
     ↓
MediaPipe – Blink & Emotion Analysis
     ↓
FaceNet – Face Embedding Extraction
     ↓
SVM Classifier – Identity Prediction
     ↓
Performance Overlay (FPS)
```

🎯 Why This Project Matters

This project goes beyond basic OpenCV face detection demos and focuses on how real systems are built:

Uses deep learning–based detection and recognition

Implements liveness verification to reduce spoofing

Designed for real-time performance

Built with a modular and extensible architecture

It reflects practical face recognition systems used in attendance systems, access control, and surveillance applications — not just academic experimentation.
