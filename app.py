import streamlit as st
import pandas as pd
import joblib
import os

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="ShifaAI", layout="wide")

# ------------------- CUSTOM CSS -------------------
st.markdown("""
<style>
/* App background */
.stApp {
    background-color: #ffffff;
    font-family: 'Segoe UI', sans-serif;
}

/* Header */
.header {
    padding: 1.5rem;
    border-radius: 12px;
    background: linear-gradient(90deg, #fce4ec, #e0f7fa);
    color: #000;
    text-align: center;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
}

.header h1 {
    font-size: 2.5rem;
    color: #e91e63;
    margin-bottom: 0.2rem;
}

.header p {
    font-size: 1.2rem;
    color: #00796b;
}

/* Buttons */
div.stButton > button {
    border-radius: 12px;
    padding: 0.65rem 1.5rem;
    font-weight: 600;
    font-size: 1rem;
    background: linear-gradient(90deg, #e91e63, #00796b);
    color: white;
    border: none;
    transition: transform 0.2s;
}
div.stButton > button:hover {
    transform: scale(1.05);
}

/* Sidebar */
.stSidebar .sidebar-content {
    background-color: #fce4ec;
    border-radius: 12px;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ------------------- HEADER -------------------
col1, col2 = st.columns([6,1])
with col1:
    st.markdown('<div class="header"><h1>ShifaAI</h1><p>AI-powered molecule screening for faster, safer, and accessible drug discovery</p></div>', unsafe_allow_html=True)
with col2:
    st.image("ShifaAi.png", width=100)

# ------------------- MODEL LOADING -------------------
MODEL_PATH = "C:/Users/KIIT/OneDrive/Desktop/hackodisha5.0/best_model_pipeline.joblib"
pipeline, model_error = None, None
if os.path.exists(MODEL_PATH):
    try:
        pipeline = joblib.load(MODEL_PATH)
    except Exception as e:
        model_error = str(e)
else:
    model_error = "❌ Model file not found. Please upload it manually."

# ------------------- FEATURE SPECS -------------------
feature_specs = {
    "age_group": {"type": "categorical", "choices": ["adult", "senior", "child", "teen"]},
    "acetaminophen": {"type": "number"},
    "acetaminophen_intensity": {"type": "number"},
    "ibuprofen": {"type": "number"},
    "ibuprofen_intensity": {"type": "number"},
    "dextromethorphan": {"type": "number"},
    "dextromethorphan_intensity": {"type": "number"},
    "guaifenesin": {"type": "number"},
    "guaifenesin_intensity": {"type": "number"},
    "phenylephrine": {"type": "number"},
    "phenylephrine_intensity": {"type": "number"},
    "chlorpheniramine": {"type": "number"},
    "chlorpheniramine_intensity": {"type": "number"},
    "n_actives": {"type": "number"},
    "total_intensity": {"type": "number"},
    "suitability_score": {"type": "number"},
}

# ------------------- HELPERS -------------------
def one_hot_encode_age(df):
    if "age_group" in df.columns:
        df = pd.get_dummies(df, columns=["age_group"], drop_first=True)
    return df

def ensure_dummy_columns(df, pipeline):
    if hasattr(pipeline, "feature_names_in_"):
        for col in pipeline.feature_names_in_:
            if col not in df.columns:
                df[col] = 0
        df = df[pipeline.feature_names_in_]
    return df

# ------------------- SIDEBAR -------------------
with st.sidebar:
    st.header("⚙ Controls")
    if model_error:
        st.error(f"Model load error: {model_error}")
    else:
        st.success("✅ Model loaded successfully!")
    st.divider()
    page = st.radio("📍 Navigate", ["🔮 Predict", "📊 Explore Data", "🧠 Model Info"], index=0)

# ------------------- PREDICTION PAGE -------------------
if page == "🔮 Predict":
    st.subheader("Enter Input Features")
    user_inputs = {}
    cols = st.columns(3)
    for i, (feature, specs) in enumerate(feature_specs.items()):
        col = cols[i % 3]
        with col:
            if specs["type"] == "categorical":
                user_inputs[feature] = st.selectbox(f"{feature}", specs["choices"])
            else:
                user_inputs[feature] = st.number_input(f"{feature}", value=0.0, step=0.1)
    predict_btn = st.button("🚀 Predict")
    if predict_btn:
        if pipeline is None:
            st.error("❌ Model not loaded.")
        else:
            X = pd.DataFrame([user_inputs])
            X = one_hot_encode_age(X)
            X = ensure_dummy_columns(X, pipeline)
            st.write("🔎 Processed Input Data:")
            st.dataframe(X)
            try:
                y_pred = pipeline.predict(X)
                st.success(f"✅ Prediction: {y_pred[0]}")
                if hasattr(pipeline, "predict_proba"):
                    y_prob = pipeline.predict_proba(X)
                    st.info(f"Class probabilities: {y_prob[0]}")
            except Exception as e:
                st.error(f"⚠ Prediction error: {e}")

# ------------------- EXPLORE DATA -------------------
elif page == "📊 Explore Data":
    st.subheader("Feature Inputs Overview")
    st.write(pd.DataFrame.from_dict(feature_specs, orient="index"))

# ------------------- MODEL INFO -------------------
elif page == "🧠 Model Info":
    st.subheader("Model Details")
    if pipeline:
        st.success("✅ Pipeline loaded")
        if hasattr(pipeline, "feature_names_in_"):
            st.write("Expected Features:", pipeline.feature_names_in_)
    else:
        st.error("❌ No model loaded")
