# 🩺 Breast Cancer Detection Using Deep Learning

An AI-powered web application for breast cancer classification using Deep Learning and Histopathology Images.

## 📌 Project Overview

This project aims to classify breast histopathology images into two categories using a trained Deep Learning model.

The system provides:
- Automated image classification.
- Prediction confidence scores.
- Model evaluation and performance metrics.
- Grad-CAM visualizations for model explainability.
- Web-based interface for interacting with the model.

## 🧠 Model Performance

The model was evaluated on a test dataset containing 51,979 samples.

| Metric | Result |
|---|---:|
| Accuracy | 83% |
| ROC-AUC | 0.9065 |
| Precision (Class 0) | 0.94 |
| Recall (Class 0) | 0.83 |
| Precision (Class 1) | 0.65 |
| Recall (Class 1) | 0.85 |

### Confusion Matrix

| | Predicted 0 | Predicted 1 |
|---|---:|---:|
| Actual 0 | 31,120 | 6,493 |
| Actual 1 | 2,156 | 12,210 |

Best classification threshold: `0.57868`

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- Deep Learning
- Convolutional Neural Networks (CNN)
- Flask
- HTML, CSS, JavaScript
- Docker
- Jupyter Notebook
- Grad-CAM

## 📂 Project Structure

```text
breast_cancer_project/
│
├── api/
│   ├── app.py
│   ├── static/
│   │   ├── gradcam_heatmap.jpg
│   │   ├── gradcam_overlay.jpg
│   │   ├── script.js
│   │   └── style.css
│   └── templates/
│       └── index.html
│
├── model/
│   └── breast_cancer_model.keras
│
├── final project.ipynb
├── requirements.txt
├── Dockerfile
└── .gitignore
```

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/gotza90/Breast-Cancer-Detection.git
cd Breast-Cancer-Detection
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
python api/app.py
```

Open your browser and navigate to the local address displayed by Flask.

## 🔍 Explainable AI – Grad-CAM

Grad-CAM is used to visualize image regions that contribute to the model's predictions, helping users better understand the model's decision process.

## ⚠️ Disclaimer

This project is developed for educational and research purposes only. It is not intended to replace professional medical diagnosis or clinical decision-making.

## 👨‍💻 Author

**Youssef Gotza**

GitHub: [@gotza90](https://github.com/gotza90)

---

⭐ If you find this project interesting, feel free to star the repository!

