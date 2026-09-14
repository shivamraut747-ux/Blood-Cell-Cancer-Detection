import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf

# Configure Streamlit Page
st.set_page_config(
    page_title="HemaVision AI - Blood Cell Cancer Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Clinical Dashboard Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2563eb;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

CLASSES = ['basophil', 'eosinophil', 'erythroblast', 'lymphocyte', 'monocyte', 'platelet']

CLINICAL_INFO = {
    'basophil': {
        'name': 'Basophil',
        'type': 'Granulocyte (White Blood Cell)',
        'reference_range': '0% - 1% (< 100 / mcL)',
        'morphology': 'Large dark purple/black coarse granules that often obscure the multi-lobed nucleus.',
        'clinical_relevance': 'Marked basophilia is a major hallmark of Chronic Myeloid Leukemia (CML) and other myeloproliferative neoplasms (MPNs). Also seen in hypersensitivity reactions.',
        'risk_level': 'Moderate / Malignancy Evaluation needed if persistently elevated'
    },
    'eosinophil': {
        'name': 'Eosinophil',
        'type': 'Granulocyte (White Blood Cell)',
        'reference_range': '1% - 4% (50 - 500 / mcL)',
        'morphology': 'Distinctive spherical reddish-orange/pink cytoplasmic granules with a bi-lobed (spectacle-shaped) nucleus.',
        'clinical_relevance': 'Markedly elevated in Hypereosinophilic Syndrome (HES), Chronic Eosinophilic Leukemia (CEL), parasitic infections, and allergic disorders.',
        'risk_level': 'Low-to-Moderate (investigate HES or systemic allergies)'
    },
    'erythroblast': {
        'name': 'Erythroblast (Nucleated Red Blood Cell - NRBC)',
        'type': 'Immature Erythroid Precursor',
        'reference_range': '0% in normal adult peripheral blood (confined to bone marrow)',
        'morphology': 'Round cell with dense, dark condensing chromatin surrounded by hemoglobinizing pink/blue cytoplasm.',
        'clinical_relevance': 'Appearance in peripheral blood (leukoerythroblastic reaction) indicates severe bone marrow stress, Acute Leukemia, Myelofibrosis, or bone marrow metastases.',
        'risk_level': 'High Clinical Significance (requires hematopathology review)'
    },
    'lymphocyte': {
        'name': 'Lymphocyte',
        'type': 'Agranulocyte (Mononuclear White Blood Cell)',
        'reference_range': '20% - 40% (1,000 - 4,000 / mcL)',
        'morphology': 'Single round or slightly indented dark purple nucleus occupying most of the cell with a thin rim of clear pale blue cytoplasm.',
        'clinical_relevance': 'Primary diagnostic cell type in Acute Lymphoblastic Leukemia (ALL), Chronic Lymphocytic Leukemia (CLL), Non-Hodgkin Lymphoma, and viral infections.',
        'risk_level': 'Key Target for ALL / CLL Screening'
    },
    'monocyte': {
        'name': 'Monocyte',
        'type': 'Agranulocyte (Largest Normal Blood Cell)',
        'reference_range': '2% - 8% (200 - 800 / mcL)',
        'morphology': 'Large cell with abundant gray-blue cytoplasm, ground-glass appearance, often with fine vacuoles and an irregular folded/kidney-shaped nucleus.',
        'clinical_relevance': 'Monocytosis is a cardinal feature of Chronic Myelomonocytic Leukemia (CMML) and Acute Myeloid Leukemia (AML - FAB M4/M5).',
        'risk_level': 'Moderate-to-High (evaluate for CMML/AML if sustained)'
    },
    'platelet': {
        'name': 'Platelet (Thrombocyte)',
        'type': 'Anucleate Cytoplasmic Fragment',
        'reference_range': '150,000 - 450,000 / mcL',
        'morphology': 'Small, disc-shaped fragments with purple granular centers derived from megakaryocytes.',
        'clinical_relevance': 'Abnormal counts or giant dysplastic platelets indicate Essential Thrombocythemia (ET), Immune Thrombocytopenia (ITP), or Myelodysplastic Syndromes (MDS).',
        'risk_level': 'Critical for Hemostasis & Bone Marrow Evaluation'
    }
}

@st.cache_resource
def load_model_by_name(model_name):
    try:
        model = tf.keras.models.load_model(model_name)
        return model
    except Exception as e:
        return None

available_models = ['Bloods.h5'] if os.path.exists('Bloods.h5') else []
default_model = 'Bloods.h5' if available_models else None


# Header
st.markdown('<div class="main-title">🔬 HemaVision AI - Blood Cell Cancer Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Clinical-grade peripheral blood smear analysis powered by Fine-Tuned EfficientNetB3 Convolutional Neural Networks</div>', unsafe_allow_html=True)

# Top Key Performance Indicators (Matching Google Colab evaluation results)
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown('<div class="metric-card"><div class="metric-val">97.50%</div><div class="metric-lbl">Test Accuracy</div></div>', unsafe_allow_html=True)
with kpi2:
    st.markdown('<div class="metric-card"><div class="metric-val">97.92%</div><div class="metric-lbl">Training Accuracy</div></div>', unsafe_allow_html=True)
with kpi3:
    st.markdown('<div class="metric-card"><div class="metric-val">96.67%</div><div class="metric-lbl">Validation Accuracy</div></div>', unsafe_allow_html=True)
with kpi4:
    st.markdown('<div class="metric-card"><div class="metric-val">0.648</div><div class="metric-lbl">Test Loss</div></div>', unsafe_allow_html=True)

st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Model Selection")
    if available_models:
        selected_model_name = st.selectbox(
            "Active AI Model:",
            available_models,
            index=available_models.index(default_model) if default_model in available_models else 0,
            help="Bloods.h5 is the fully trained model with 97.5% test accuracy."
        )
        model = load_model_by_name(selected_model_name)
        st.success(f"✅ Active: **{selected_model_name}**")
        st.caption("Architecture: **EfficientNetB3 + Regularized Head**")
    else:
        st.error("⚠️ No model file (.h5) found in project.")
        model = None
        selected_model_name = None

    st.markdown("---")
    st.header("🧪 Sample Image Gallery")
    st.caption("Test immediately with sample cells from the dataset:")
    
    dataset_dir = r"X:\archive\bloodcells_dataset"
    sample_options = ["-- None (Upload your own) --"]
    sample_paths = {}
    
    if os.path.exists(dataset_dir):
        for cls in CLASSES:
            folder = os.path.join(dataset_dir, cls)
            if os.path.exists(folder):
                files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                if files:
                    chosen = files[1] if len(files) > 1 else files[0]
                    label = f"{cls.capitalize()} ({chosen})"
                    sample_options.append(label)
                    sample_paths[label] = os.path.join(folder, chosen)
    
    selected_sample = st.selectbox("Choose a sample cell:", sample_options)
    
    st.markdown("---")
    st.header("🎯 Confidence Calibration")
    sharpness_mode = st.radio(
        "Prediction Mode:",
        ["High Confidence (90%+ Target)", "Standard Softmax"],
        index=0,
        help="High Confidence uses temperature scaling (T=0.55) to sharpen output probabilities into the 90-99% range."
    )
    temp = 0.55 if sharpness_mode == "High Confidence (90%+ Target)" else 1.0

    st.markdown("---")
    st.markdown("### 📚 Target Classes")
    for cls in CLASSES:
        st.markdown(f"- **{cls.capitalize()}**")

# Navigation Tabs
tab_live, tab_history, tab_metrics, tab_dataset = st.tabs([
    "🔍 Live Cell Classifier & Diagnosis",
    "📈 Training & Accuracy Curves",
    "🧮 Confusion Matrix & Metrics",
    "🖼️ Training Dataset Explorer"
])

# TAB 1: Live Cell Classifier
with tab_live:
    col_left, col_right = st.columns([1.1, 1.3], gap="large")

    image_to_process = None
    image_source_name = ""

    with col_left:
        st.subheader("📷 Microscopic Blood Smear Input")
        uploaded_file = st.file_uploader(
            "Drag & drop a peripheral blood cell image (.jpg, .png, .jpeg)",
            type=["jpg", "jpeg", "png"],
            help="Upload a standard 100x oil immersion microscopic field image"
        )
        
        if uploaded_file is not None:
            image_to_process = Image.open(uploaded_file).convert("RGB")
            image_source_name = uploaded_file.name
        elif selected_sample != "-- None (Upload your own) --":
            path = sample_paths[selected_sample]
            image_to_process = Image.open(path).convert("RGB")
            image_source_name = os.path.basename(path)
            st.info(f"Loaded sample: **{image_source_name}**")
            
        if image_to_process is not None:
            st.image(image_to_process, caption=f"Analyzed Smear Field: {image_source_name}", use_container_width=True)
        else:
            st.info("👆 Upload an image or select a sample from the left sidebar to start diagnostic analysis.")

    with col_right:
        st.subheader("📊 Diagnostic AI Prediction")
        
        if image_to_process is not None:
            if model is None:
                st.error("Model weights file (`Bloods.h5`) is required to run inference.")
            else:
                with st.spinner("Analyzing cell morphology & nuclear chromatin..."):
                    if uploaded_file is not None:
                        uploaded_file.seek(0)
                        img_keras = tf.keras.utils.load_img(uploaded_file, target_size=(224, 224))
                    else:
                        img_keras = tf.keras.utils.load_img(path, target_size=(224, 224))
                    img_array = tf.keras.utils.img_to_array(img_keras)
                    img_batch = np.expand_dims(img_array, axis=0)
                    
                    raw_preds = model.predict(img_batch, verbose=0)[0]
                    # Apply Temperature Calibration
                    if temp != 1.0:
                        logits = np.log(raw_preds + 1e-7) / temp
                        exp_logits = np.exp(logits - np.max(logits))
                        preds = exp_logits / np.sum(exp_logits)
                    else:
                        preds = raw_preds
                        
                    top_idx = int(np.argmax(preds))
                    top_class = CLASSES[top_idx]
                    top_confidence = float(preds[top_idx]) * 100
                    
                    badge_color = "#16a34a" if top_confidence >= 80 else ("#eab308" if top_confidence >= 50 else "#dc2626")
                    st.markdown(f"""
                    <div style="background-color: {badge_color}; color: white; padding: 14px 22px; border-radius: 10px; margin-bottom: 15px;">
                        <div style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em;">Predicted Cell Type</div>
                        <div style="font-size: 1.9rem; font-weight: 700;">{top_class.upper()}</div>
                        <div style="font-size: 1.15rem;">Confidence: <b>{top_confidence:.2f}%</b></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if top_confidence < 60:
                        st.warning("⚠️ **Low Confidence Notice:** The image may contain multiple adjacent cells (a doublet) or atypical staining.")
                    
                    prob_df = pd.DataFrame({
                        'Cell Type': [c.capitalize() for c in CLASSES],
                        'Confidence (%)': [float(p * 100) for p in preds],
                        'Probability': [float(p) for p in preds]
                    }).sort_values('Confidence (%)', ascending=True)
                    
                    fig = px.bar(
                        prob_df,
                        x='Confidence (%)',
                        y='Cell Type',
                        orientation='h',
                        text=prob_df['Confidence (%)'].apply(lambda x: f"{x:.2f}%"),
                        color='Confidence (%)',
                        color_continuous_scale=['#cbd5e1', '#60a5fa', '#2563eb', '#1d4ed8'],
                        range_x=[0, 100],
                        title="Class Probability Distribution"
                    )
                    fig.update_layout(
                        height=290,
                        margin=dict(l=10, r=20, t=40, b=10),
                        showlegend=False,
                        coloraxis_showscale=False
                    )
                    fig.update_traces(textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)

    if image_to_process is not None and model is not None:
        st.markdown("---")
        st.subheader(f"📋 Clinical Hematology Insights: {top_class.capitalize()}")
        info = CLINICAL_INFO.get(top_class, {})
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**Cell Classification:**<br>{info.get('type', 'N/A')}", unsafe_allow_html=True)
        with c2:
            st.markdown(f"**Normal Adult Range:**<br>{info.get('reference_range', 'N/A')}", unsafe_allow_html=True)
        with c3:
            st.markdown(f"**Clinical Status:**<br>{info.get('risk_level', 'N/A')}", unsafe_allow_html=True)
            
        st.markdown(f"**🔬 Morphological Characteristics:** {info.get('morphology', 'N/A')}")
        st.markdown(f"**🩺 Diagnostic & Cancer Relevance:** {info.get('clinical_relevance', 'N/A')}")

# TAB 2: Training History Curves
with tab_history:
    st.subheader("📈 Training and Validation History Curves")
    st.caption("Convergence and epoch progression recorded during the deep learning training session on Google Colab GPU:")
    
    history_img_path = os.path.join("assets", "training_history.png")
    if os.path.exists(history_img_path):
        st.image(history_img_path, caption="Training vs Validation Loss (Left) and Accuracy (Right) with Best Epoch Markers", use_container_width=True)
    else:
        st.info("Training history graph asset not found.")
        
    st.markdown("""
    ### 🔍 Analysis of Training Convergence:
    - **Loss Progression:** Loss steadily drops from $> 10.0$ down to **0.59 (Training)** and **0.62 (Validation)**, demonstrating smooth gradient descent with the Adamax optimizer.
    - **Accuracy Convergence:** Training accuracy rapidly reaches **> 97%** within the initial epochs and stabilizes at **97.9%**, confirming effective feature extraction through the pre-trained EfficientNetB3 backbone.
    - **Generalization:** The close alignment between training loss and validation loss confirms absence of severe overfitting.
    """)

# TAB 3: Confusion Matrix & Metrics
with tab_metrics:
    st.subheader("🧮 Confusion Matrix & Detailed Performance Metrics")
    st.caption("Evaluated on 870 unseen test images across all 6 blood cell classes:")
    
    col_cm, col_rep = st.columns([1.1, 1.2], gap="large")
    
    with col_cm:
        cm_img_path = os.path.join("assets", "confusion_matrix.png")
        if os.path.exists(cm_img_path):
            st.image(cm_img_path, caption="Confusion Matrix Heatmap (True vs Predicted Labels)", use_container_width=True)
        else:
            st.info("Confusion matrix asset not found.")
            
    with col_rep:
        st.markdown("### 📋 Classification Report (Test Set)")
        report_data = {
            'Cell Class': ['Basophil', 'Eosinophil', 'Erythroblast', 'Lymphocyte', 'Monocyte', 'Platelet'],
            'Precision': ['97%', '97%', '95%', '96%', '95%', '100%'],
            'Recall': ['94%', '99%', '93%', '93%', '98%', '100%'],
            'F1-Score': ['96%', '98%', '94%', '94%', '97%', '100%'],
            'Test Samples (Support)': [102, 238, 120, 107, 128, 175]
        }
        rep_df = pd.DataFrame(report_data)
        st.dataframe(rep_df, use_container_width=True, hide_index=True)
        
        st.markdown("""
        - **Overall Test Accuracy:** **97.50%** (848 / 870 images correctly classified)
        - **Macro Average F1-Score:** **96.5%**
        - **Platelets:** 100% Precision & 100% Recall (Zero misclassifications)
        - **Eosinophils:** 99% Recall (Extremely sensitive detection of granules)
        """)

# TAB 4: Training Dataset Explorer
with tab_dataset:
    st.subheader("🖼️ Training Dataset Sample Explorer")
    st.caption("Sample microscopic field images from the Kaggle Blood Cancer Dataset:")
    
    samples_img_path = os.path.join("assets", "dataset_samples.png")
    if os.path.exists(samples_img_path):
        st.image(samples_img_path, caption="Sample Blood Cell Smears from Training Data", use_container_width=True)
    else:
        st.info("Dataset samples asset not found.")
        
    st.markdown("""
    ### 🔬 About the Dataset:
    - **Total Images:** 17,092 high-resolution microscopic blood smear images.
    - **Resolution:** Cropped and standardized to $224 \times 224$ pixels, 3 color channels (RGB).
    - **Staining:** Standard Romanowsky / Giemsa Wright staining under $100\times$ oil immersion objective.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94a3b8; font-size: 0.85rem;'>"
    "HemaVision AI • Developed by Shivam Raut • Built with TensorFlow, EfficientNetB3 & Streamlit • For Clinical Research & Education"
    "</div>",
    unsafe_allow_html=True
)
