# 🩺 Meiyalagu AI
### Intelligent Skin Disease & Skin Cancer Risk Screening System

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3.2-lightgrey.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 Overview

**Meiyalagu AI** is an AI-powered web application designed for **early screening** of skin diseases and skin cancer risk. 

Users can upload or capture a photograph of a skin lesion. The system:
1. **Checks image quality** (blur, brightness, framing).
2. **Classifies the lesion** using a MobileNetV2-based deep learning model (trained on the HAM10000 dataset).
3. **Assesses cancer risk** (High/Low concern).
4. **Explains the prediction** using Grad-CAM heatmaps.
5. **Stores screening history** and tracks risk trends over time.

> ⚠️ **Disclaimer:** This is an **AI-assisted screening tool** and **does not** replace a clinical dermatologist. It is designed for educational and decision-support purposes only.

---

## ✨ Key Features

- 📸 **Image Upload**: Drag-and-drop or browse for skin lesion images.
- ✅ **Image Quality Check**: Rejects blurry, dark, or overexposed images before analysis.
- 🧠 **AI Classification**: Predicts lesion categories (Nevus, Melanoma, BCC, etc.).
- 🔥 **Explainable AI (Grad-CAM)**: Visual heatmap highlighting the region the AI focused on.
- 🚦 **Cancer Risk Screening**: Categorizes results as "High Concern" or "Lower Concern".
- 👤 **Patient Tracking**: Enter patient names to maintain personalized history.
- 📊 **Trend Detection**: Alerts if a patient's risk level has escalated since their last screening.
- 💾 **Screening History**: Stores all past analyses in an SQLite database.
- 🖥️ **Responsive UI**: Clean, medical-grade interface that works on desktop and mobile.

---

## 🛠️ Tech Stack

| Component          | Technology                                     |
| ------------------ | ---------------------------------------------- |
| **Language**       | Python 3.10                                    |
| **Deep Learning**  | TensorFlow / Keras (MobileNetV2)               |
| **Backend**        | Flask                                          |
| **Image Processing**| OpenCV / Pillow                               |
| **Explainable AI** | Grad-CAM                                       |
| **Database**       | SQLite                                         |
| **Frontend**       | HTML5, CSS3, Vanilla JavaScript                |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- Git
- (Optional) Virtual environment

### Installation

1. **Clone the repository**
   git clone https://github.com/rathishakarthikeyan-cmd/meiyalagu-ai.git
   cd meiyalagu-ai

2. **Create and activate a virtual environment** (Recommended)
   - **Windows**:
     python -m venv venv
     venv\Scripts\activate
   - **Mac/Linux**:
     python3 -m venv venv
     source venv/bin/activate

3. **Install dependencies**
   pip install -r requirements.txt

4. **Run the application**
   python app.py

5. **Open your browser** and navigate to:
   http://127.0.0.1:5000

---

## 🧠 Model Training (Coming Soon)

Currently, the application uses a **dummy predictor** for demonstration purposes. 

---

## 📁 Project Structure

meiyalagu_ai/
├── app.py                  # Flask main application
├── database.py             # SQLite database operations
├── model_utils.py          # Image quality check & dummy predictor
├── grad_cam.py             # Grad-CAM heatmap generator
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignored files (venv, uploads, etc.)
├── templates/
│   └── index.html          # Main frontend page
├── static/
│   ├── style.css           # Styling
│   └── script.js           # Frontend logic
├── uploads/                # User-uploaded images (ignored by Git)
├── instance/               # Database files (ignored by Git)
└── README.md               # This file

