# Fake Vision – Detecting AI-Generated Images with CNN

## 🧠 Project Overview

**Fake Vision** is a deep learning project designed to classify whether an image is **real** or **AI-generated** using a custom-built Convolutional Neural Network (CNN). This project uses the [CIFAKE dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) which contains realistic fake and real images.

## 🔍 Objective

To build a binary image classifier that distinguishes between AI-generated (fake) and real images with high accuracy and minimal overfitting.

---

## 🧰 Tech Stack

- Python
- TensorFlow / Keras
- CNN (Convolutional Neural Networks)
- Matplotlib, Seaborn
- Scikit-learn
- Kaggle Notebooks

---

## 🗃️ Dataset

- **Name**: CIFAKE: Real and AI-Generated Synthetic Images
- **Source**: [Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
- 2 Classes: `REAL`, `FAKE`
- Used 5000 images per class (subset for faster training)

---

## 🧪 Model Architecture

- 3 × Conv2D layers (ReLU + BatchNorm + MaxPooling)
- Dropout (0.3, 0.4)
- Dense (128 + 1 sigmoid)
- Optimizer: Adam
- Loss Function: Binary Crossentropy

---

## 📈 Results

- ✅ **Test Accuracy**: ~88%
- 🧠 **Model Evaluation**:
  - Confusion Matrix
  - Precision / Recall / F1-Score
  - Misclassified Image Visualization

---

## 📊 Visualization

- Training vs Validation Accuracy/Loss plots
- Confusion Matrix
- Misclassified samples display

---

## 📦 Output Files

- `best_model.keras` – Saved best model using ModelCheckpoint
- `plot_metrics()` – Accuracy/Loss Graphs
- `confusion_matrix()` – Heatmap of misclassifications

---

## 🤖 Future Work

- Apply transfer learning (ResNet, EfficientNet)
- Improve generalization with more augmentation
- Deploy as a REST API using Flask

---

## 👤 Author

**Sri Charan Pydisetti**  
Final Year B.Tech Student  
Deep Learning | AI Enthusiast | Problem Solver  
[GitHub](https://github.com/Sricharan9761)

---
