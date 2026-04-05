# 🩺 HealthGuard — AI-Driven Diabetes Onset Prediction

> A deep learning web application that predicts the risk of diabetes onset from 8 clinical measurements — built with **Keras**, **Keras Tuner**, and **Streamlit**.

---

## 📌 What This Project Does

You enter a patient's basic health data (like glucose level, BMI, age, etc.) and the app tells you:
- ✅ **Low risk** — the model thinks this person is unlikely to have diabetes
- ⚠️ **High risk** — the model thinks this person may have diabetes

It also shows a **confidence percentage** so you know how sure the model is.

---

## 🖥️ Live Demo (Screenshot)

The app has a modern dark medical UI with:
- 8 input fields for patient data
- A big **Predict** button
- A color-coded result card with a confidence bar
- Input summary metrics

---

## 📊 Dataset

- **Name:** Pima Indians Diabetes Dataset
- **Source:** Originally from the National Institute of Diabetes and Digestive and Kidney Diseases
- **Size:** 768 rows × 9 columns
- **Target:** `Outcome` — `1` = Diabetic, `0` = Not Diabetic

### Features used:

| Feature | Description |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration (mg/dL) |
| BloodPressure | Diastolic blood pressure (mmHg) |
| SkinThickness | Triceps skinfold thickness (mm) |
| Insulin | 2-hour serum insulin (µU/mL) |
| BMI | Body Mass Index (kg/m²) |
| DiabetesPedigreeFunction | Family history score |
| Age | Age in years |

---

## 🧠 How the Model Was Built

### Step 1 — Exploratory Data Analysis (EDA)

The notebook `ML/notebook/eda.ipynb` covers:
- Data loading and inspection
- Detecting that features like **Glucose, BloodPressure, SkinThickness, Insulin, BMI** contain `0` values which are biologically impossible — treated as **missing values**
- Filling missing values using **group median** (grouped by `Outcome`)
- Capping outliers using **IQR (Interquartile Range)** method for `Insulin`, `BMI`, and `DiabetesPedigreeFunction`
- Feature scaling using **StandardScaler**

### Step 2 — Keras Tuner Hyperparameter Optimization

Instead of guessing the best network architecture, we ran **4 Keras Tuner experiments** in sequence:

| Tuner | What it searched |
|---|---|
| Tuner 1 | Best **optimizer** (Adam, SGD, RMSprop…) |
| Tuner 2 | Best number of **hidden layers** |
| Tuner 3 | Best number of **nodes per layer** |
| Tuner 4 | **All of the above combined** — final fine-tuning |

Running them sequentially (1 → 2 → 3 → 4) progressively narrows the search space so the final model generalizes well without overfitting.

### Step 3 — Final Model Architecture

```
Input (8 features)
    ↓
Dense Layer (tuned nodes, ReLU activation)
    ↓
Dense Output (1 node, Sigmoid activation)
    ↓
Output: Probability of diabetes (0.0 – 1.0)
```

- **Loss function:** Binary Crossentropy
- **Trained for:** 1000 epochs
- **Threshold:** Probability ≥ 0.5 → Diabetic

---

## 🚀 How to Run This Project Locally

### Prerequisites
- Python 3.10+
- Git

### 1. Clone the repository

```bash
git clone https://github.com/7vaibhav31/HealthGuard-AI-Driven-Diabetes-Onset-Prediction.git
cd HealthGuard-AI-Driven-Diabetes-Onset-Prediction
```

### 2. Create and activate a virtual environment

```bash
# Create venv
python -m venv venv

# Activate (Windows)
.\venv\Scripts\Activate.ps1

# Activate (Mac/Linux)
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
python -m streamlit run main.py
```

Then open **`http://localhost:8501`** in your browser.

> **Note:** `model.keras` and `scaler.pkl` are already included in the repository — no need to retrain.

---

## 📁 Project Structure

```
HealthGuard-AI-Driven-Diabetes-Onset-Prediction/
│
├── main.py               ← Streamlit web application
├── model.keras           ← Trained Keras neural network
├── scaler.pkl            ← Fitted StandardScaler (for input preprocessing)
├── requirements.txt      ← Python dependencies
│
└── ML/
    ├── data/
    │   └── diabetes.csv  ← Pima Indians Diabetes dataset
    └── notebook/
        └── eda.ipynb     ← Full EDA + model training notebook
```

---

## ⚙️ Preprocessing at Inference Time

When you enter values in the app, the following steps happen **before** the model sees your input:

1. **Zero imputation** — If you enter `0` for Glucose, BloodPressure, SkinThickness, Insulin, or BMI (which are impossible values), the app replaces them with the average of the non-diabetic and diabetic group medians from the training data.
2. **IQR capping** — Extreme values for Insulin, BMI, and DiabetesPedigreeFunction are clipped to the bounds computed during training.
3. **Scaling** — All 8 features are scaled using the `StandardScaler` that was fitted on training data (loaded from `scaler.pkl`).

This ensures the input is processed **exactly the same way** as the training data.

---

## 📦 Dependencies

```
tensorflow
keras
keras-tuner
scikit-learn
pandas
numpy
streamlit
joblib
matplotlib
seaborn
```

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It is **not a substitute** for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical decisions.

---

## 👨‍💻 Author

**Vaibhav Sharma**  
B.Tech — Computer Science & Engineering (AI/ML)  
📧 m.7vansh31@gmail.com

---

## ⭐ If you found this useful, give it a star!
