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
    page_icon="??",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Medical / Clinical Dashboard Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .prediction-badge {
        display: inline-block;
        font-size: 1.3rem;
        font-weight: 700;
        padding: 0.4rem 1.2rem;
        border-radius: 50px;
        color: white;
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        margin-bottom: 0.8rem;
    }
    .clinical-title {
        font-weight: 700;
        color: #0f172a;
        font-size: 1.1rem;
        margin-top: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Define Classes (Alphabetical order as indexed during training)
CLASSES = ['basophil', 'eosinophil', 'erythroblast', 'lymphocyte', 'monocyte', 'platelet']

# Clinical Knowledge Base for Hematologic Interpretation
CLINICAL_INFO = {
    'basophil': {
        'name': 'Basophil',
        'type': 'Granulocyte (White Blood Cell)',
        'reference_range': '0% - 1% (< 100 / mcL)',
        'morphology': 'Large dark purple/black granules that frequently obscure the multi-lobed nucleus.',
        'clinical_relevance': 'Marked basophilia is a classic hallmark of Chronic Myeloid Leukemia (CML) and other myeloproliferative neoplasms (MPNs). Also elevated in severe hypersensitivity reactions.',
        'risk_level': 'Moderate / Malignancy Evaluation needed if persistently elevated'
    },
    'eosinophil': {
        'name': 'Eosinophil',
        'type': 'Granulocyte (White Blood Cell)',
        'reference_range': '1% - 4% (50 - 500 / mcL)',
        'morphology': 'Distinctive coarse, spherical reddish-orange/pink cytoplasmic granules with a bi-lobed (spectacle-shaped) nucleus.',
        'clinical_relevance': 'Markedly elevated in Hypereosinophilic Syndrome (HES), Chronic Eosinophilic Leukemia (CEL), parasitic infections, and allergic disorders.',
        'risk_level': 'Low-to-Moderate (investigate HES or systemic allergies)'
    },
    'erythroblast': {
        'name': 'Erythroblast (Nucleated Red Blood Cell - NRBC)',
        'type': 'Immature Erythroid Precursor',
        'reference_range': '0% in normal adult peripheral blood (confined to bone marrow)',
        'morphology': 'Round cell with dense, dark, condensing chromatin surrounded by hemoglobinizing pink/blue cytoplasm.',
        'clinical_relevance': 'Appearance in peripheral blood (leukoerythroblastic reaction) indicates severe bone marrow stress, Acute Leukemia, Myelofibrosis, or bone marrow metastases.',
        'risk_level': 'High Clinical Significance (requires hematopathology review)'
    },
    'lymphocyte': {
        'name': 'Lymphocyte',
        'type': 'Agranulocyte (Mononuclear White Blood Cell)',
        'reference_range': '20% - 40% (1,000 - 4,000 / mcL)',
        'morphology': 'Single round or slightly indented dark purple nucleus occupying most of the cell with a thin rim of clear pale blue cytoplasm.',
        'clinical_relevance': 'Crucial diagnostic cell type in Acute Lymphoblastic Leukemia (ALL), Chronic Lymphocytic Leukemia (CLL), Non-Hodgkin Lymphoma, and viral infections (mononucleosis).',
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
        'morphology': 'Small, disc-shaped cell fragments with purple granular centers derived from megakaryocytes.',
        'clinical_relevance': 'Abnormal counts or giant dysplastic platelets indicate Essential Thrombocythemia (ET), Immune Thrombocytopenia (ITP), or Myelodysplastic Syndromes (MDS).',
        'risk_level': 'Critical for Hemostasis & Bone Marrow Evaluation'
    }
}

# Cached Model Loading Function
@st.cache_resource
def load_trained_model():
    candidates = [
        "Bloods_High_Confidence.h5",
        "Bloods.h5",
        os.path.join(os.path.dirname(__file__), "Bloods_High_Confidence.h5"),
        os.path.join(os.path.dirname(__file__), "Bloods.h5"),
        r"X:\Bloods.h5"
    ]
    
    for path in candidates:
        if os.path.exists(path):
            try:
                model = tf.keras.models.load_model(path)
                return model, os.path.basename(path)
            except Exception as e:
                continue
    return None, None

model, model_filename = load_trained_model()

# Header
st.markdown('<div class="main-title">?? HemaVision AI - Blood Cell Cancer Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Clinical-grade peripheral blood smear analysis powered by Fine-Tuned EfficientNetB3 Convolutional Neural Networks</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("?? System Status")
    if model is not None:
        st.success(f"? Model Loaded: **{model_filename}**")
        st.caption("Architecture: **EfficientNetB3 + Dense Regularized Head**")
    else:
        st.error("?? Model file not found.")
        st.info("Place `Bloods_High_Confidence.h5` or `Bloods.h5` into your project directory.")

    st.markdown("---")
    st.header("?? Sample Image Gallery")
    st.caption("Quickly test with sample cells from the local dataset:")
    
    # Check for local dataset folder
    dataset_dir = r"X:\archive\bloodcells_dataset"
    sample_options = ["-- None (Upload your own) --"]
    sample_paths = {}
    
    if os.path.exists(dataset_dir):
        for cls in CLASSES:
            folder = os.path.join(dataset_dir, cls)
            if os.path.exists(folder):
                files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                if files:
                    # Pick sample images with typical single cell morphology
                    chosen = files[1] if len(files) > 1 else files[0]
                    label = f"{cls.capitalize()} ({chosen})"
                    sample_options.append(label)
                    sample_paths[label] = os.path.join(folder, chosen)
    
    selected_sample = st.selectbox("Choose a sample cell:", sample_options)
    
    st.markdown("---")
    st.markdown("### ?? Reference Classes")
    for cls in CLASSES:
        st.markdown(f"- **{cls.capitalize()}**")

# Main View
col_left, col_right = st.columns([1.1, 1.3], gap="large")

image_to_process = None
image_source_name = ""

with col_left:
    st.subheader("?? Microscopic Blood Smear Input")
    
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
        st.image(image_to_process, caption=f"Analyzed Smear Field: {image_source_name}", use_column_width=True)
    else:
        st.info("?? Upload an image or select a sample from the left sidebar to start diagnostic analysis.")

with col_right:
    st.subheader("?? Diagnostic AI Prediction")
    
    if image_to_process is not None:
        if model is None:
            st.error("Model weights file (`Bloods.h5`) is required to run inference.")
        else:
            with st.spinner("Analyzing cell morphology & nuclear chromatin..."):
                # Preprocess
                img_resized = image_to_process.resize((224, 224))
                img_array = tf.keras.utils.img_to_array(img_resized)
                img_batch = np.expand_dims(img_array, axis=0)
                
                # Predict
                preds = model.predict(img_batch, verbose=0)[0]
                top_idx = int(np.argmax(preds))
                top_class = CLASSES[top_idx]
                top_confidence = float(preds[top_idx]) * 100
                
                # Top Prediction Badge
                badge_color = "#16a34a" if top_confidence >= 80 else ("#eab308" if top_confidence >= 50 else "#dc2626")
                st.markdown(f"""
                <div style="background-color: {badge_color}; color: white; padding: 12px 20px; border-radius: 8px; margin-bottom: 15px;">
                    <div style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em;">Predicted Cell Type</div>
                    <div style="font-size: 1.8rem; font-weight: 700;">{top_class.upper()}</div>
                    <div style="font-size: 1.1rem;">Confidence: <b>{top_confidence:.2f}%</b></div>
                </div>
                """, unsafe_allow_html=True)
                
                if top_confidence < 60:
                    st.warning("?? **Low Confidence Notice:** The image may contain multiple adjacent cells (a doublet) or non-standard staining.")
                
                # Probability Distribution Chart
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
                    height=280,
                    margin=dict(l=10, r=20, t=40, b=10),
                    showlegend=False,
                    coloraxis_showscale=False
                )
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)

# Bottom Section: Detailed Clinical Analysis
if image_to_process is not None and model is not None:
    st.markdown("---")
    st.subheader(f"?? Clinical Hematology Insights: {top_class.capitalize()}")
    
    info = CLINICAL_INFO.get(top_class, {})
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"**Cell Classification:**<br>{info.get('type', 'N/A')}", unsafe_allow_html=True)
    with c2:
        st.markdown(f"**Normal Adult Range:**<br>{info.get('reference_range', 'N/A')}", unsafe_allow_html=True)
    with c3:
        st.markdown(f"**Clinical Status:**<br>{info.get('risk_level', 'N/A')}", unsafe_allow_html=True)
        
    st.markdown(f"**?? Morphological Characteristics:** {info.get('morphology', 'N/A')}")
    st.markdown(f"**?? Diagnostic & Cancer Relevance:** {info.get('clinical_relevance', 'N/A')}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94a3b8; font-size: 0.85rem;'>"
    "HemaVision AI  Developed by Shivam Raut  Built with TensorFlow, EfficientNetB3 & Streamlit  For Clinical Research & Education"
    "</div>",
    unsafe_allow_html=True
)
