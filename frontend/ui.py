import streamlit as st
import requests
import json

# Page Configuration
st.set_page_config(page_title="Sleep Disorder Prediction System", page_icon="🌙", layout="centered")

# Main Title and Description
st.title("🌙 Sleep Disorder Diagnostic System")
st.markdown("### Please enter your health metrics below to predict potential sleep disorders.")
st.markdown("---")

# --- Form Section (Moved to Main Area) ---
st.header("📋 Patient Information")

# Create two columns to organize the inputs better (User Experience Improvement)
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 10, 90, 32)
    occupation = st.selectbox("Occupation", [
        "Software Engineer", "Doctor", "Sales Representative", "Teacher", 
        "Nurse", "Engineer", "Accountant", "Scientist", "Lawyer", 
        "Salesperson", "Manager"
    ])
    sleep_duration = st.slider("Sleep Duration (hours)", 4.0, 10.0, 7.0, 0.1)
    quality_of_sleep = st.slider("Quality of Sleep (1-10)", 1, 10, 7)
    physical_activity = st.slider("Physical Activity Level (mins/day)", 0, 100, 40)

with col2:
    stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
    bmi_category = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese"])
    blood_pressure = st.text_input("Blood Pressure (e.g., 120/80)", "120/80")
    heart_rate = st.number_input("Heart Rate (bpm)", 60, 120, 70)
    daily_steps = st.number_input("Daily Steps", 0, 20000, 5000)

# Construct the data dictionary for the API
input_data = {
    "gender": gender,
    "age": age,
    "occupation": occupation,
    "sleep_duration": sleep_duration,
    "quality_of_sleep": quality_of_sleep,
    "physical_activity_level": physical_activity,
    "stress_level": stress_level,
    "bmi_category": bmi_category,
    "blood_pressure": blood_pressure,
    "heart_rate": heart_rate,
    "daily_steps": daily_steps
}

st.markdown("---")

# --- Prediction Section (Bottom) ---
# Center the button using columns to make it look nicer
_, mid_col, _ = st.columns([1, 2, 1])

with mid_col:
    predict_btn = st.button("🚀 Start Prediction", type="primary", use_container_width=True)

if predict_btn:
    with st.spinner("Model is analyzing data..."):
        try:
            # Localhost refers to the container itself here
            api_url = "http://127.0.0.1:8000/invocations" 
            
            response = requests.post(api_url, json=input_data)
            
            if response.status_code == 200:
                result = response.json()
                prediction = result.get("prediction", "Unknown")
                
                st.success("✅ Prediction Complete!")
                
                # Display Results
                st.subheader(f"Diagnostic Result: {prediction}")
                
                if prediction == "None":
                    st.info("Congratulations! No significant sleep disorder risk detected. Keep up the healthy lifestyle! (Source: Model)")
                elif prediction == "Insomnia":
                    st.warning("⚠️ Warning: Risk of Insomnia detected. It is recommended to consult a doctor or improve your sleep schedule.")
                elif prediction == "Sleep Apnea":
                    st.error("🚨 Warning: Risk of Sleep Apnea detected. Please seek medical attention as soon as possible.")
            else:
                st.error(f"❌ Prediction Failed: {response.text}")
                
        except Exception as e:
            st.error(f"❌ Unable to connect to backend service: {e}")

# Footer
st.markdown("---")
st.caption("Powered by MLOps Pipeline & Streamlit")