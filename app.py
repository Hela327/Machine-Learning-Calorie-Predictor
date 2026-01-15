import streamlit as st
import numpy as np
import joblib

# =============================
# Load trained model and scaler
# =============================
model = joblib.load("calories_model.pkl")
scaler = joblib.load("scaler.pkl")

# =============================
# Page configuration
# =============================
st.set_page_config(
    page_title="Gym Calories Burn Predictor",
    layout="centered"
)

# =============================
# Background image + styling
# =============================
st.markdown(
    """
    <style>
    /* Full page background image */
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1571019613454-1cb2f99b2d8b");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    /* Main content card */
    .block-container {
        background-color: rgba(0, 0, 0, 0.75);
        padding: 2.5rem;
        border-radius: 18px;
        max-width: 900px;
    }

    /* App title */
    .title-text {
        font-size: 42px;
        font-weight: bold;
        color: #00FFAA;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle-text {
        font-size: 18px;
        color: #E0E0E0;
        text-align: center;
        margin-bottom: 35px;
    }

    /* Section header style */
    .section-header {
        background: linear-gradient(90deg, #00FFAA, #00C896);
        color: #000000;
        padding: 10px 16px;
        border-radius: 10px;
        font-size: 20px;
        font-weight: bold;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    /* Result box */
    .result-box {
        background-color: rgba(0, 0, 0, 0.9);
        padding: 22px;
        border-radius: 14px;
        text-align: center;
        font-size: 28px;
        color: #00FFAA;
        font-weight: bold;
        margin-top: 25px;
        border: 2px solid #00FFAA;
    }

    /* Improve label visibility */
    label {
        color: #F1F1F1 !important;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =============================
# Title
# =============================
st.markdown('<div class="title-text">🏋️‍♂️ Gym Calories Burn Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle-text">Estimate calories burned using body metrics and workout activity</div>',
    unsafe_allow_html=True
)

# =============================
# Body Information Section
# =============================
st.markdown('<div class="section-header">🧍 Body Information</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age (years)", 1, 100, 25)
    height = st.number_input("Height (cm)", 100, 220, 170)
    bmi = st.number_input("BMI", 10.0, 50.0, 22.0)

with col2:
    body_temp = st.number_input("Body Temperature (°C)", 35.0, 42.0, 37.0)
    body_fat = st.number_input("Body Fat Percentage (%)", 5.0, 60.0, 20.0)
    rest_hr = st.number_input("Resting Heart Rate (bpm)", 40, 120, 70)

# =============================
# Workout Information Section
# =============================
st.markdown('<div class="section-header">🔥 Workout & Activity Details</div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:
    heart_rate = st.number_input("Workout Heart Rate (bpm)", 50, 200, 90)
    steps = st.number_input("Steps Count", 0, 50000, 8000)

with col4:
    sleep = st.number_input("Sleep Hours", 0.0, 12.0, 7.0)
    water = st.number_input("Water Intake (Liters)", 0.0, 10.0, 2.5)
    active_min = st.number_input("Daily Active Minutes", 0, 300, 60)

# =============================
# Prediction Section
# =============================
st.markdown("---")

if st.button("🔥 Calculate Calories Burned"):

    gender_val = 0 if gender == "Male" else 1

    # IMPORTANT: Order must match training feature order
    input_data = np.array([[
        gender_val,
        age,
        height,
        heart_rate,
        body_temp,
        bmi,
        steps,
        sleep,
        water,
        body_fat,
        rest_hr,
        active_min
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.markdown(
        f'<div class="result-box">🔥 {prediction[0]:.2f} kcal burned</div>',
        unsafe_allow_html=True
    )
