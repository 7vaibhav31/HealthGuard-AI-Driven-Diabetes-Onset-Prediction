"""
main.py  –  HealthGuard: AI‑Driven Diabetes Onset Prediction
============================================================
Streamlit application that:
  1. Loads model.keras and scaler.pkl from the project root.
  2. Accepts 8 clinical inputs from the user.
  3. Applies the same preprocessing used during training:
       • Replaces physiologically impossible zeros with group‑median estimates.
       • Scales features with the fitted StandardScaler.
  4. Runs the Keras model and displays the prediction.

Run with:
    streamlit run main.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import tensorflow as tf

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HealthGuard – Diabetes Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS – modern dark medical UI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Google Font ──────────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global resets ───────────────────────────────────────────────────── */
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

    /* ── App background ──────────────────────────────────────────────────── */
    .stApp {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b2a3d 60%, #0d1b2a 100%);
        color: #e8edf3;
    }

    /* ── Sidebar ─────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: rgba(13, 27, 42, 0.95);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    [data-testid="stSidebar"] h2 {
        color: #38bdf8;
        font-weight: 700;
    }

    /* ── Main header card ────────────────────────────────────────────────── */
    .header-card {
        background: linear-gradient(120deg, #1e3a5f 0%, #1a4e7e 100%);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(56, 189, 248, 0.25);
        text-align: center;
    }
    .header-card h1 {
        font-size: 2.6rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .header-card p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 0.6rem;
        margin-bottom: 0;
    }

    /* ── Section titles ──────────────────────────────────────────────────── */
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #38bdf8;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
        margin-top: 1.4rem;
        border-left: 4px solid #38bdf8;
        padding-left: 0.75rem;
    }

    /* ── Input labels ────────────────────────────────────────────────────── */
    label { color: #cbd5e1 !important; font-weight: 500 !important; }

    /* ── Number inputs ───────────────────────────────────────────────────── */
    .stNumberInput input {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(56,189,248,0.3) !important;
        border-radius: 10px !important;
        color: #e8edf3 !important;
        font-size: 1rem !important;
        transition: border 0.2s;
    }
    .stNumberInput input:focus {
        border: 1.5px solid #38bdf8 !important;
        box-shadow: 0 0 0 3px rgba(56,189,248,0.15) !important;
    }

    /* ── Predict button ──────────────────────────────────────────────────── */
    div[data-testid="stButton"] > button {
        width: 100%;
        padding: 0.9rem 2rem;
        border-radius: 14px;
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        background: linear-gradient(90deg, #0ea5e9, #818cf8);
        border: none;
        color: #fff;
        cursor: pointer;
        box-shadow: 0 4px 20px rgba(14,165,233,0.35);
        transition: transform 0.15s, box-shadow 0.15s;
        margin-top: 1.5rem;
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(14,165,233,0.5);
    }
    div[data-testid="stButton"] > button:active { transform: translateY(0); }

    /* ── Result cards ────────────────────────────────────────────────────── */
    .result-high {
        background: linear-gradient(135deg, #450a0a 0%, #7f1d1d 100%);
        border: 1.5px solid #f87171;
        border-radius: 18px;
        padding: 2rem 2.5rem;
        margin-top: 1.5rem;
        box-shadow: 0 8px 32px rgba(248,113,113,0.25);
        text-align: center;
    }
    .result-low {
        background: linear-gradient(135deg, #052e16 0%, #14532d 100%);
        border: 1.5px solid #4ade80;
        border-radius: 18px;
        padding: 2rem 2.5rem;
        margin-top: 1.5rem;
        box-shadow: 0 8px 32px rgba(74,222,128,0.2);
        text-align: center;
    }
    .result-icon { font-size: 3rem; margin-bottom: 0.4rem; }
    .result-title {
        font-size: 1.7rem;
        font-weight: 800;
        margin: 0.2rem 0 0.5rem;
    }
    .result-sub { color: #cbd5e1; font-size: 0.95rem; margin: 0; }

    /* ── Confidence bar ──────────────────────────────────────────────────── */
    .conf-bar-bg {
        background: rgba(255,255,255,0.08);
        border-radius: 8px;
        height: 12px;
        overflow: hidden;
        margin-top: 1.2rem;
    }
    .conf-bar-fill-high {
        height: 100%;
        background: linear-gradient(90deg, #f87171, #ef4444);
        border-radius: 8px;
        transition: width 0.6s ease;
    }
    .conf-bar-fill-low {
        height: 100%;
        background: linear-gradient(90deg, #4ade80, #22c55e);
        border-radius: 8px;
        transition: width 0.6s ease;
    }
    .conf-label {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 0.4rem;
        text-align: center;
    }

    /* ── Info / warning boxes ────────────────────────────────────────────── */
    .info-box {
        background: rgba(56,189,248,0.08);
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        color: #94a3b8;
        font-size: 0.88rem;
        margin-top: 1rem;
    }

    /* ── Divider ─────────────────────────────────────────────────────────── */
    hr { border-color: rgba(255,255,255,0.08); }

    /* ── Metrics ─────────────────────────────────────────────────────────── */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 0.8rem 1rem;
        border: 1px solid rgba(255,255,255,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Load artefacts
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artefacts():
    model_path  = os.path.join(os.path.dirname(__file__), "model.keras")
    scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")

    if not os.path.exists(model_path):
        st.error(
            "❌  **model.keras not found.**  "
            "Please run the save cells in your notebook (eda.ipynb) to export the model."
        )
        st.stop()
    if not os.path.exists(scaler_path):
        st.error(
            "❌  **scaler.pkl not found.**  "
            "Please run the save cells in your notebook (eda.ipynb) to export the scaler."
        )
        st.stop()

    model  = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_artefacts()

# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing helper
# These medians were computed on the *full* dataset (by outcome group) in
# eda.ipynb. We reproduce them here for inference-time imputation.
# ─────────────────────────────────────────────────────────────────────────────

# Imputation medians: {feature: {0: median_for_nondiabetic, 1: median_for_diabetic}}
# Values are read directly from the dataset in the notebook.
IMPUTATION_MEDIANS = {
    "Glucose":       {0: 107.0,  1: 140.0},
    "BloodPressure": {0: 70.0,   1: 74.0},
    "SkinThickness": {0: 27.0,   1: 32.0},
    "Insulin":       {0: 102.5,  1: 169.5},
    "BMI":           {0: 30.1,   1: 34.3},
}

def preprocess_input(pregnancies, glucose, blood_pressure,
                     skin_thickness, insulin, bmi,
                     dpf, age):
    """
    Mirror the notebook preprocessing:
    1. Replace impossible zero values with group‑median estimates
       (we use the mean of both group medians as an unbiased prior
        since we cannot know the true label at inference time).
    2. Apply IQR outlier capping (thresholds from the dataset).
    3. Scale with the fitted StandardScaler.
    """

    # Step 1 – Impute zeros
    def safe(val, feature):
        if val == 0:
            m0 = IMPUTATION_MEDIANS[feature][0]
            m1 = IMPUTATION_MEDIANS[feature][1]
            return (m0 + m1) / 2.0  # neutral prior
        return float(val)

    glucose       = safe(glucose,       "Glucose")
    blood_pressure= safe(blood_pressure,"BloodPressure")
    skin_thickness= safe(skin_thickness,"SkinThickness")
    insulin       = safe(insulin,       "Insulin")
    bmi           = safe(bmi,           "BMI")

    # Step 2 – IQR capping (pre‑computed bounds from the full dataset)
    IQR_CAPS = {
        "Insulin":                  (  0.0, 318.0),
        "BMI":                      ( 18.2,  47.39),
        "DiabetesPedigreeFunction": (  0.078, 1.2),
    }
    def cap(val, lo, hi):
        return float(np.clip(val, lo, hi))

    insulin = cap(insulin, *IQR_CAPS["Insulin"])
    bmi     = cap(bmi,     *IQR_CAPS["BMI"])
    dpf     = cap(dpf,     *IQR_CAPS["DiabetesPedigreeFunction"])

    # Step 3 – Scale
    raw = np.array([[pregnancies, glucose, blood_pressure,
                     skin_thickness, insulin, bmi, dpf, age]], dtype=float)
    scaled = scaler.transform(raw)
    return scaled


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="header-card">
        <h1>🩺 HealthGuard</h1>
        <p>AI‑Driven Diabetes Onset Prediction &nbsp;·&nbsp; Keras Neural Network</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – About & tips
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ℹ️ About")
    st.markdown(
        """
        **HealthGuard** uses a deep neural network trained on the
        Pima Indians Diabetes dataset to estimate the probability of
        a patient having diabetes based on 8 clinical measurements.

        ---
        ### 📋 How to Use
        1. Enter the patient's clinical data on the right.
        2. Click **Predict**.
        3. Review the result and confidence score.

        ---
        ### ⚠️ Disclaimer
        This tool is for **educational and research purposes only**
        and is **not a substitute** for professional medical advice.
        """
    )
    st.markdown("---")
    st.markdown("### 🤖 Model Info")
    st.metric("Architecture",  "Dense(32,relu) → Dense(1,sigmoid)")
    st.metric("Trained Epochs", "1000")
    st.metric("Training Dataset", "Pima Indians Diabetes (768 rows)")

# ─────────────────────────────────────────────────────────────────────────────
# Input form
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Patient Information</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0, max_value=20, value=1, step=1,
        help="Number of times pregnant",
    )
    glucose = st.number_input(
        "Glucose (mg/dL)",
        min_value=0, max_value=300, value=120, step=1,
        help="Plasma glucose concentration (2h oral glucose tolerance test). Enter 0 if unknown.",
    )
    blood_pressure = st.number_input(
        "Blood Pressure (mmHg)",
        min_value=0, max_value=150, value=70, step=1,
        help="Diastolic blood pressure. Enter 0 if unknown.",
    )

with col2:
    skin_thickness = st.number_input(
        "Skin Thickness (mm)",
        min_value=0, max_value=120, value=20, step=1,
        help="Triceps skin fold thickness. Enter 0 if unknown.",
    )
    insulin = st.number_input(
        "Insulin (µU/mL)",
        min_value=0, max_value=846, value=80, step=1,
        help="2‑hour serum insulin. Enter 0 if unknown.",
    )
    bmi = st.number_input(
        "BMI (kg/m²)",
        min_value=0.0, max_value=80.0, value=32.0, step=0.1,
        help="Body mass index. Enter 0 if unknown.",
        format="%.1f",
    )

with col3:
    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.000, max_value=2.500, value=0.471, step=0.001,
        help="A function which scores likelihood of diabetes based on family history.",
        format="%.3f",
    )
    age = st.number_input(
        "Age (years)",
        min_value=1, max_value=120, value=33, step=1,
        help="Age in years.",
    )

# Info note about zero imputation
st.markdown(
    '<div class="info-box">'
    '💡 <strong>Tip:</strong> Enter <strong>0</strong> for any measurement that is unavailable. '
    'The model will automatically replace it with a statistically derived estimate '
    'based on the training dataset.'
    '</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────
if st.button("🔬 Predict Diabetes Risk", use_container_width=True):
    with st.spinner("Analysing clinical data …"):
        X_scaled = preprocess_input(
            pregnancies, glucose, blood_pressure,
            skin_thickness, insulin, bmi, dpf, age,
        )
        prob = float(model.predict(X_scaled, verbose=0)[0][0])
        is_diabetic = prob >= 0.5

    # ── Result card ──────────────────────────────────────────────────────────
    pct = int(round(prob * 100))
    bar_width = pct

    if is_diabetic:
        st.markdown(
            f"""
            <div class="result-high">
                <div class="result-icon">⚠️</div>
                <div class="result-title" style="color:#f87171;">High Diabetes Risk Detected</div>
                <p class="result-sub">
                    The model estimates a <strong>{pct}%</strong> probability of diabetes onset.
                </p>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill-high" style="width:{bar_width}%;"></div>
                </div>
                <p class="conf-label">Confidence: {pct}%</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-low">
                <div class="result-icon">✅</div>
                <div class="result-title" style="color:#4ade80;">Low Diabetes Risk</div>
                <p class="result-sub">
                    The model estimates only a <strong>{pct}%</strong> probability of diabetes onset.
                </p>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill-low" style="width:{bar_width}%;"></div>
                </div>
                <p class="conf-label">Confidence (risk level): {pct}%</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Metrics breakdown ─────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Input Summary</div>', unsafe_allow_html=True)

    # Display the raw input values entered by the user
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pregnancies",   f"{pregnancies}")
    m2.metric("Glucose",       f"{glucose} mg/dL")
    m3.metric("Blood Pressure",f"{blood_pressure} mmHg")
    m4.metric("Skin Thickness",f"{skin_thickness} mm")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Insulin",       f"{insulin} µU/mL")
    m6.metric("BMI",           f"{bmi:.1f} kg/m²")
    m7.metric("DPF",           f"{dpf:.3f}")
    m8.metric("Age",           f"{age} yrs")

    st.markdown(
        '<div class="info-box">'
        '⚠️ <strong>Medical Disclaimer:</strong> This prediction is generated by an AI model '
        'for research and educational use only. It should <strong>not</strong> be used as a '
        'substitute for professional medical diagnosis or advice. Please consult a healthcare '
        'professional for any medical concerns.'
        '</div>',
        unsafe_allow_html=True,
    )
