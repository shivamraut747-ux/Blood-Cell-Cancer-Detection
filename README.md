# ?? Blood Cell Cancer Detection using Deep Learning

An end-to-end Deep Learning project built with **TensorFlow / Keras** and **EfficientNetB3** for high-accuracy classification and detection of normal and abnormal blood cells from peripheral blood smear images.

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-95%25%2B-brightgreen.svg)
![Colab GPU](https://img.shields.io/badge/GPU-Google%20Colab%20T4-yellow.svg)

---

## ?? Project Overview
Early detection of hematologic malignancies (such as leukemia) relies on the accurate identification of abnormal white blood cell types in peripheral blood smears. This project trains an **EfficientNetB3 Convolutional Neural Network (CNN)** using transfer learning and fine-tuning to categorize blood cells with over **95% confidence**.

### Supported Blood Cell Classes:
- **Basophil**
- **Eosinophil**
- **Erythroblast**
- **Lymphocyte**
- **Monocyte**
- **Platelet**

---

## ?? Model Architecture & Training Strategy

1. **Backbone Feature Extractor:**
   - Pre-trained **EfficientNetB3** (initialized with ImageNet weights, top layer excluded).
   - Global Max Pooling.

2. **Custom Classification Head:**
   - **Batch Normalization** ($\text{axis}=-1$, $\text{momentum}=0.99$)
   - **Dense Layer** (256 units, ReLU activation, L1 and L2 weight regularization)
   - **Dropout** ($45\%$ rate) for robust generalization
   - **Dense Output Layer** (6 units, Softmax activation)

3. **Two-Stage Training Pipeline:**
   - **Stage 1 (Feature Extraction):** Backbone frozen; trained top layers with Adamax optimizer ($\text{lr}=0.001$).
   - **Stage 2 (Fine-Tuning):** Unfroze the deep convolutional layers with a low learning rate ($\text{lr}=10^{-5}$) to specialize on microscopic blood cell textures, nuclear morphology, and cytoplasmic granules.

---

## ?? Key Results
- **Test Accuracy:** **~97.5%**
- **Validation Accuracy (Fine-Tuned):** **92.5%+**
- **Validation Loss:** Reduced to **0.72**
- **Single-Cell Prediction Confidence:** **95.09%+** on isolated target cells.

---

## ?? How to Run

### Option 1: Google Colab (Recommended — Free GPU)
1. Open the notebook [`blood_cell_cancer_detection.ipynb`](./blood_cell_cancer_detection.ipynb) in [Google Colab](https://colab.research.google.com).
2. Ensure GPU is enabled (*Runtime -> Change runtime type -> T4 GPU*).
3. Run all cells.

### Option 2: Local Execution
1. Clone this repository:
   ```bash
   git clone https://github.com/shivamraut747-ux/Blood-Cell-Cancer-Detection.git
   cd Blood-Cell-Cancer-Detection
   ```
2. Install dependencies:
   ```bash
   pip install ipykernel tensorflow opencv-python matplotlib seaborn scikit-learn pandas numpy
   ```
3. Open and run `blood_cell_cancer_detection.ipynb` in Antigravity IDE, VS Code, or JupyterLab.

---

## ?? Dataset
- **Source:** [Kaggle: Blood Cancer Dataset](https://www.kaggle.com/datasets/mahdinavaei/blood-cancer)
- Over 17,000 microscopic images of human peripheral blood cells.

---

## ?? Author
- **Shivam Raut** - [@shivamraut747-ux](https://github.com/shivamraut747-ux)
