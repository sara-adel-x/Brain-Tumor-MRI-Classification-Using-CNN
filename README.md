# 🧠 Brain Tumor MRI Classification Using CNN

### AI-Powered Medical Decision Support System

An AI-powered medical imaging system for automatic brain tumor classification from MRI scans using a Convolutional Neural Network (CNN). The project integrates Explainable AI (Grad-CAM) with an interactive Streamlit web application, enabling healthcare professionals to classify brain tumors, visualize the model's decision-making process, and generate downloadable medical reports.

---

## 📌 Project Overview

Brain tumors are among the most critical neurological diseases, where early and accurate diagnosis can significantly improve treatment outcomes. Manual MRI interpretation is time-consuming and highly dependent on expert radiologists.

This project presents an AI-based decision support system capable of classifying brain MRI images into four categories:

- 🟢 No Tumor
- 🔴 Glioma
- 🟠 Meningioma
- 🟣 Pituitary Tumor

The system combines Deep Learning, Explainable AI, and an intuitive graphical interface to make AI-assisted diagnosis accessible, interpretable, and user-friendly.

---

# 🚀 Features

- 🧠 Brain MRI Classification using CNN
- 📂 Four-Class Classification
  - Glioma
  - Meningioma
  - Pituitary Tumor
  - No Tumor
- 🔥 Grad-CAM Heatmap Visualization
- 📊 Confidence Score
- 📈 Class Probability Distribution
- 👤 Patient Information Form
- 📄 PDF Medical Report Generation
- 📁 JSON Report Export
- 🖼 Downloadable Heatmap Image
- ⚡ Interactive Streamlit GUI

---

# 📂 Dataset

**Brain Tumor MRI Dataset**

Source:

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

### Dataset Statistics

| Item | Count |
|------|------:|
| Total Images | 7200 |
| Training Images | 5600 |
| Testing Images | 1600 |
| Classes | 4 |

### Classes

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

---

# ⚙️ Data Preparation

The MRI images undergo several preprocessing steps before training:

- Train / Validation / Test Split
- Image Resizing (150 × 150)
- Pixel Normalization (Rescale = 1/255)
- Data Augmentation
  - Rotation
  - Width Shift
  - Height Shift
  - Zoom
- Efficient Data Loading using ImageDataGenerator

These preprocessing techniques improve the model's ability to generalize while reducing overfitting.

---

# 🧠 CNN Model

### Architecture

Our custom CNN architecture consists of:

- Conv2D (32 Filters)
- MaxPooling
- Conv2D (64 Filters)
- MaxPooling
- Conv2D (128 Filters)
- MaxPooling
- Flatten
- Dense (128)
- Dropout (0.3)
- Softmax Output Layer (4 Classes)

### Training Configuration

Optimizer:

- Adam

Loss Function:

- Categorical Crossentropy

Callbacks:

- EarlyStopping
- ModelCheckpoint

Task:

Multi-Class Image Classification

Framework:

TensorFlow / Keras

---

# 📊 Results

### Model Performance

- ✅ Test Accuracy: **94%**
- Multi-Class Classification
- Confusion Matrix
- Classification Report
- Precision
- Recall
- F1-Score

The model achieved strong generalization performance while maintaining stable validation accuracy through data augmentation and regularization techniques.

---

# 🔥 Explainable AI (Grad-CAM)

To improve model transparency, Grad-CAM (Gradient-weighted Class Activation Mapping) was integrated into the system.

Instead of producing only a prediction, the model highlights the regions of the MRI scan that contributed most to its decision.

This enables healthcare professionals to better understand and verify the AI's predictions, making the system more interpretable and trustworthy.

---

# 💻 Streamlit Web Application

An interactive Streamlit interface was developed to simplify the prediction process.

### Features

- MRI Image Upload
- Patient Information Form
- Brain Tumor Prediction
- Confidence Score
- Confidence Gauge
- Class Probability Distribution
- Grad-CAM Heatmap Visualization
- Processing Time
- Device Information (CPU/GPU)
- Downloadable PDF Report
- Downloadable JSON Report
- Downloadable Heatmap Image

---

# 📄 Generated Report Includes

Each generated report contains:

- Patient Information
- Hospital Information
- MRI Scan Date
- Predicted Tumor Type
- Confidence Score
- Risk Level
- Class Probabilities
- Grad-CAM Visualization
- AI Recommendation
- Model Information
- Processing Time
- Medical Disclaimer

Reports can be exported as:

- 📄 PDF
- 📁 JSON
- 🖼 Heatmap Image

---

# 📊 System Workflow

```text
MRI Image
      │
      ▼
Image Preprocessing
      │
      ▼
CNN Classification
      │
      ▼
Prediction + Confidence Score
      │
      ▼
Grad-CAM Heatmap
      │
      ▼
Medical Report Generation
      │
      ▼
Export (PDF / JSON / Heatmap)
```

---

# 🛠 Technologies Used

- Python
- TensorFlow
- Keras
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Pillow
- Grad-CAM
- ReportLab
- JSON

---

# 📷 Application Preview

### Home Page

<img width="1472" height="1037" alt="image" src="https://github.com/user-attachments/assets/5531f531-bf9d-4f04-a831-cd19cfb1718e" />

---

### Prediction Results

<img width="1650" height="975" alt="image" src="https://github.com/user-attachments/assets/f3846a73-c481-4b6c-a875-7b810f3dbb73" />

---

### Tumor Information

<img width="1426" height="622" alt="tumor info" src="https://github.com/user-attachments/assets/a205c431-6d67-45e9-b4a4-d7a6f3cc8bd7" />

---

### Grad-CAM Visualization

<img width="1627" height="695" alt="Screenshot 2026-08-06 235005" src="https://github.com/user-attachments/assets/bdd72f14-6acf-4397-98ba-bcdf631167eb" />


---

### Generated PDF Report

<img width="847" height="1011" alt="image" src="https://github.com/user-attachments/assets/558d20ee-9ca8-4a42-ba00-b1e593092285" />

---

### Evaluation Scores

<img width="1392" height="545" alt="evaluation scores" src="https://github.com/user-attachments/assets/b4688d9b-297e-4871-ba6a-2eda997fe87c" />


---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/YourUsername/Brain-Tumor-MRI-Classification.git
```

Navigate to the project

```bash
cd Brain-Tumor-MRI-Classification
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 🎯 Future Improvements

- Implement Transfer Learning (EfficientNet, ResNet)
- Support DICOM Medical Images
- Tumor Segmentation before Classification
- Cloud Deployment
- Hospital Information System Integration
- Larger Multi-Center MRI Datasets
- Advanced Explainability Methods (Grad-CAM++, Score-CAM)
- Mobile Application Support

---

# ⚠️ Disclaimer

This project is intended for educational and research purposes only.

The AI-generated predictions should not be considered a medical diagnosis. Clinical decisions should always be made by qualified healthcare professionals.

---

# 👥 Team

Developed by

- **Sara Adel** 
- **Eman Rashad** (@emanrashad1912)
- **John Fawzy** (@johnfms)
- **Seif Nour** (@gholamseif)
---

## ⭐ If you found this project useful, don't forget to star the repository!
