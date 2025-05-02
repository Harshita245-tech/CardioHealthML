import streamlit as st
import numpy as np
import pickle
import os

st.title("💓 Heart Disease Predictor")
st.markdown("Fill the details to predict the risk of heart disease.")

# Load the trained model safely
model_path = "lr_model.pkl"
if not os.path.exists(model_path):
    st.error("Model file not found. Please upload `lr_model.pkl` to the app directory.")
    st.stop()

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# User inputs
age = st.number_input("Age", min_value=1, max_value=120, value=60)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
chol = st.number_input("Serum Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["Yes", "No"])
restecg = st.selectbox("Resting ECG results", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.number_input("ST depression induced by exercise", min_value=0.0, max_value=6.0, value=1.0, format="%.1f")
slope = st.selectbox("Slope of peak exercise ST segment", [0, 1, 2])
ca = st.selectbox("Number of major vessels (0-3) colored by fluoroscopy", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)", [1, 2, 3])

# Process input
input_data = np.array([[
    age,
    1 if sex == "Male" else 0,
    cp,
    trestbps,
    chol,
    1 if fbs == "Yes" else 0,
    restecg,
    thalach,
    1 if exang == "Yes" else 0,
    oldpeak,
    slope,
    ca,
    thal
]])

# Prediction
if st.button("🔍 Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.markdown("✅ **Prediction: Heart Disease: Yes**")
        st.markdown(f"📊 **Probability: {probability:.2f}**")
        st.warning("🚨 **Suggestion: High risk – Seek immediate medical attention!**")
    else:
        st.markdown("✅ **Prediction: Heart Disease: No**")
        st.markdown(f"📊 **Probability: {probability:.2f}**")
        st.success("👍 **Suggestion: Low risk – Keep monitoring your health.**")
