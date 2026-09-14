# 🔬 Blood Cell Classifier

An end-to-end Deep Learning system built with **TensorFlow / Keras**, **EfficientNetB3**, and **Streamlit** for high-accuracy classification, clinical analysis, and live interactive detection of blood cell malignancies from microscopic blood smears.

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-95%25%2B-brightgreen.svg)
![Colab GPU](https://img.shields.io/badge/GPU-Google%20Colab%20T4-yellow.svg)

---

## 📌 Project Overview
Early detection of hematologic malignancies (such as Acute Lymphoblastic Leukemia, CML, and myelodysplastic syndromes) relies on rapid and accurate categorization of white blood cell types in peripheral blood smears. 

This project provides:
1. **Fine-Tuned EfficientNetB3 CNN:** A deep learning model achieving **95%+ confidence** on single blood cell classifications.
2. **Interactive Clinical Web Dashboard:** A web interface to drag-and-drop blood smear images, receive sub-second diagnostic classification, view class probability distributions, and review clinical hematological implications.

### Supported Blood Cell Classes:
- **Basophil** (Indicator in CML / Myeloproliferative disorders)
- **Eosinophil** (Allergic & Hypereosinophilic Syndromes)
- **Erythroblast** (Nucleated RBC, marker for severe marrow stress/leukemia)
- **Lymphocyte** (Key diagnostic cell for ALL / CLL / Lymphoma)
- **Monocyte** (Marker for CMML / AML)
- **Platelet** (Essential for thrombocythemia and hemostasis evaluation)

---

## 🖥️ Interactive Web Dashboard (HemaVision AI)

The interactive dashboard allows users to analyze blood cell images locally in real time.

### Features:
- **Drag & Drop Upload:** Upload any .jpg, .jpeg, or .png blood smear image.
- **Instant AI Diagnosis:** Live prediction badge with color-coded confidence levels.
- **Probability Breakdown:** Interactive Plotly horizontal bar chart showing probabilities across all 6 classes.
- **Clinical Insights:** Morphological breakdown, normal reference ranges, and diagnostic significance for leukemia.
- **Built-in Sample Gallery:** Test with sample blood cells in 1 click without needing your own files.

### How to Launch Dashboard:
`ash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the Streamlit dashboard
streamlit run app.py
`
The dashboard will open automatically in your browser at http://localhost:8501.

---

## 🧠 Model Architecture & Methodology

1. **Backbone Feature Extractor:**
   - Pre-trained **EfficientNetB3** (ImageNet weights, top layer excluded).
   - Global Max Pooling.

2. **Custom Classification Head:**
   - **Batch Normalization**
   - **Dense Layer** (256 units, ReLU activation, L1 and L2 weight regularization)
   - **Dropout** (45% rate) for regularization
   - **Dense Output Layer** (6 units, Softmax activation)

3. **Two-Stage Training Pipeline:**
   - **Stage 1 (Feature Extraction):** Backbone frozen; trained dense head with Adamax.
   - **Stage 2 (Fine-Tuning):** Unfroze deep convolutional layers with a low learning rate (1e-5) to adapt to microscopic cellular textures and nuclear chromatin patterns.

---

## 📊 Key Results
- **Test Accuracy:** **~97.5%**
- **Validation Accuracy (Fine-Tuned):** **92.5%+**
- **Single-Cell Prediction Confidence:** **95.09%+** on isolated target cells.

---

## 📁 Dataset
- **Source:** [Kaggle: Blood Cancer Dataset](https://www.kaggle.com/datasets/mahdinavaei/blood-cancer)
- Over 17,000 microscopic images of human peripheral blood cells.

---

## 👤 Author
- **Shivam Raut** - [@shivamraut747-ux](https://github.com/shivamraut747-ux)
