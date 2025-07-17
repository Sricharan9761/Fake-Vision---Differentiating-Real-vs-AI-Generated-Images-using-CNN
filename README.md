# Fake-Vision---Differentiating-Real-vs-AI-Generated-Images-using-CNN
# 🧠 Fake Vision – Differentiating Real vs AI Generated Images using CNN

Fake Vision is a deep learning project designed to classify images as either **real** or **AI-generated** using a Convolutional Neural Network (CNN). With the rapid rise of generative models like DALL·E, Midjourney, and GANs, the ability to distinguish between authentic and synthetic images is increasingly important for media integrity and digital forensics.

---

## 📌 Project Objectives

- Build a CNN model to accurately classify real vs AI-generated images.
- Use the **CIFAKE** dataset with balanced classes.
- Implement preprocessing, augmentation, and regularization.
- Evaluate performance using classification metrics and visual inspection.

---

## 📂 Dataset

- **Dataset Used:** [CIFAKE Dataset](https://github.com/peterbarkacs/CIFake)
- **Classes:** 
  - `Real`: Natural, human-taken images.
  - `Fake`: AI-generated images (e.g., via StyleGAN, DALL·E).
- **Total Samples:** 10,000 (5,000 real, 5,000 fake)

---

## 🔧 Tech Stack & Tools

- Python
- TensorFlow / Keras
- NumPy, OpenCV, Matplotlib
- Google Colab (for model training)

---

## 🏗️ Model Architecture

- Convolutional layers (3×)
- Max Pooling layers
- Flatten
- Dense layers
- Dropout
- Final output: **Softmax** (binary classification)

---

## 🧪 Evaluation Metrics

| Metric      | Score |
|-------------|-------|
| Accuracy    | 88%   |
| Precision   | ~87%  |
| Recall      | ~89%  |
| F1-Score    | ~88%  |

We also analyzed:
- **Confusion matrix**
- **Misclassified samples**
- **Grad-CAM** visualizations (optional future work)

---

## 🧠 Key Features

- Custom-built CNN for binary image classification.
- Data Augmentation (rotation, flip, shift) to reduce overfitting.
- Early Stopping and Dropout regularization.
- Visual validation using correctly and incorrectly classified examples.

---

## 📈 Sample Results

> Real image correctly classified ✅  
> Fake image misclassified ❌  
> Obvious AI artifacts detected in some false positives.

![Example image](https://via.placeholder.com/400x200?text=Sample+Image+Here)  
(*Replace with real output if needed*)

---

## 💡 Future Enhancements

- Use **Vision Transformers (ViT)** for better context understanding.
- Implement **real-time detection** with OpenCV and webcam feed.
- Extend to **deepfake video classification** using 3D CNNs.
- Host a demo using **Streamlit** or **Flask**.

---

## 🤝 Team Members

- **Pydisetti Sri Charan**  
  - Role: Model Design, Data Processing, Training, Evaluation  
  - [GitHub](https://github.com/Sricharan9761)

---

## 📄 Report

For a detailed report on methodology, experiments, and results, refer to the [Final Report PDF](./B2_75_DL_Finalreport.pdf).

---

## 🧾 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

- CIFAKE Dataset creators: Peter Barkacs et al.
- Keras & TensorFlow community
