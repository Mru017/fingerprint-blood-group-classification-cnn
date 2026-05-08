# Fingerprint Blood Group Classification using CNN
# Fingerprint Blood Group Classification using CNN

## 📌 Project Overview
This project presents a deep learning-based biometric classification system for predicting blood groups using fingerprint images. The system utilizes a Convolutional Neural Network (CNN) developed with PyTorch to analyze fingerprint patterns and classify blood groups.

The project includes image preprocessing, CNN model training, evaluation, and deployment through a Gradio web interface for real-time prediction.

---

## 🚀 Features
- Fingerprint image preprocessing and normalization
- CNN-based image classification using PyTorch
- Real-time blood group prediction
- Interactive Gradio deployment interface
- Model evaluation using confusion matrix and classification report
- Automated model saving using `.pth` checkpoint files

---

## 🛠️ Technologies Used
- Python
- PyTorch
- CNN (Convolutional Neural Network)
- Gradio
- NumPy
- Matplotlib
- Scikit-learn
- PIL (Python Imaging Library)

---

## 📂 Dataset
The dataset consists of fingerprint images categorized into multiple blood group classes:
- A+
- A-
- B+
- B-
- AB+
- AB-
- O+
- O-

The dataset was divided into:
- 80% Training Data
- 20% Testing Data

---

## 🧠 Model Architecture
The CNN model contains:
- Convolution Layers
- ReLU Activation
- Max Pooling Layers
- Fully Connected Layers
- Softmax Output Layer

---

## 📊 Model Performance
| Metric | Value |
|---|---|
| Model Type | CNN |
| Framework | PyTorch |
| Accuracy Achieved | 78.21% |
| Deployment | Gradio |

---

## ▶️ How to Run the Project

### 1️⃣ Install Required Libraries
```bash
pip install -r requirements.txt

### 2️⃣ Train the Model
python cnn_pytorch.py

Run the Gradio App
python app.py

🌐 Deployment
The project is deployed using Gradio for real-time fingerprint image prediction.
Upload a fingerprint image and the model predicts the corresponding blood group.

📁 Project Structure
blood_dataset/│├── app.py├── cnn_pytorch.py├── fastcnn_model.pth├── requirements.txt├── confusion_matrix.png├── train/└── test/

🔮 Future Improvements


Improve classification accuracy using larger datasets


Implement transfer learning models


Reduce class confusion between positive and negative blood groups


Deploy as a cloud-hosted web application


Develop mobile-friendly prediction interface



📌 Conclusion
This project demonstrates the application of deep learning and computer vision techniques in biometric-based blood group classification. The CNN model successfully performs fingerprint image analysis and real-time prediction through an interactive deployment interface.


