# BrainScan – Unified Explainable AI for Brain MRI Classification

🧠 Python • Deep Learning • Explainable AI • Medical Imaging

---

## 🚀 Project Overview

**BrainScan** is an intelligent deep learning system that automatically analyzes **brain MRI scans** and classifies them into four neurological categories using a **single unified model**.

It helps support early diagnosis by combining **multi-disease detection** with **visual explanations**, making AI predictions easier to understand and trust in clinical settings.

The system can classify MRI scans into:

🧠 Healthy  
🧩 Alzheimer’s Disease  
🧬 Multiple Sclerosis  
🧠 Brain Tumor  

---

## 🧠 How It Works

1️⃣ **Input MRI Scan**  
Brain MRI images are provided to the system for analysis.

2️⃣ **Deep Learning Classification**  
A lightweight **MobileNetV2-based CNN** (transfer learning) predicts the disease class.

3️⃣ **Explainable AI (Grad-CAM)**  
Grad-CAM heatmaps highlight important brain regions influencing the prediction, such as:
- Hippocampal atrophy (Alzheimer’s)
- White matter lesions (MS)
- Tumor boundaries (Brain Tumor)

---

## ⚡ Features

✅ Unified multi-disease classification using a single model  
✅ Explainable predictions with Grad-CAM heatmaps  
✅ Lightweight and efficient (CPU-friendly)  
✅ Trained using transfer learning  
✅ Designed for clinical interpretability  

---

## 📊 Dataset

- **Multi-Class Neurological Disorder (MCND) Dataset**
- Source: Kaggle  
- Total MRI images: **16,224**
- Classes: Healthy, Alzheimer’s, MS, Brain Tumor  

🔗 Dataset link:  
https://www.kaggle.com/datasets/alifatahi/multi-class-neurological-disorder-mcnd-dataset

*(Dataset not included in this repository due to size constraints.)*

---

## 🛠 Model & Training

- Model: MobileNetV2-based CNN  
- Framework: TensorFlow / Keras  
- Total parameters: ~2.42M  
- Trainable parameters: 164,484  
- Optimizer: Adam  
- Loss: Categorical Cross-Entropy  
- Validation Accuracy: **87.58%**


---

## 🛠 Installation & Setup

1. **Clone the repository:**

```bash
git clone https://github.com/anushkaverse/BrainScan.git
cd BrainScan
````

2. **Create a virtual environment (optional but recommended):**

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Run the notebook:**

```bash
notebooks/testing_brainscan.ipynb
```

---

## 📂 Project Structure

```
BrainScan/
│
├─ model/
│  └─ final_brain_multi_disease_model.keras
├─ notebooks/
│  └─ testing_brainscan.ipynb
├─ requirements.txt
├─ README.md
└─ LICENSE
```
---

## 📣 Future Improvements

* External multi-hospital validation
* Disease staging (especially for Alzheimer’s)
* Federated learning for privacy-preserving training

---

## 📜 License

This project is licensed under the **MIT License** – see [LICENSE](LICENSE) for details.

---

## 👋 Author

Developed by **Anushka Sharma** 
