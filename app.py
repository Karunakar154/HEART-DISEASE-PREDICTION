import streamlit as st
import pickle
import numpy as np

# --- LOAD THE TRAINED MODEL ---
try:
    model = pickle.load(open("model.pkl", "rb"))
except FileNotFoundError:
    st.error("Model file 'model.pkl' not found. Please train a model and save it in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred loading the model: {e}")
    st.stop()

# --- PAGE CONFIG ---
st.set_page_config(page_title="Heart Disease Prediction", page_icon="🫀", layout="wide")

# --- HEADER ---
st.markdown("""
# 🫀 10-Year Heart Disease Risk Predictor
Predict the probability of Coronary Heart Disease based on patient data.
""")

st.divider()

# --- INPUT LAYOUT ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧍‍♂️ Patient Info & Habits")
    male = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])
    age = st.number_input("Age", 20, 100, 52)
    education = st.selectbox("Education Level (1-4)", [1, 2, 3, 4])
    currentSmoker = st.selectbox("Current Smoker (0=No, 1=Yes)", [0, 1])
    cigsPerDay = st.number_input("Cigarettes Per Day", 0, 100, 0)

with col2:
    st.subheader("🩺 Medical History & Vitals")
    BPMeds = st.selectbox("On BP Medication (0=No, 1=Yes)", [0, 1])
    prevalentStroke = st.selectbox("Had Previous Stroke (0=No, 1=Yes)", [0, 1])
    prevalentHyp = st.selectbox("Has Hypertension (0=No, 1=Yes)", [0, 1])
    diabetes = st.selectbox("Has Diabetes (0=No, 1=Yes)", [0, 1])

st.divider()
st.subheader("⚕️ Clinical Measurements")
col3, col4, col5 = st.columns(3)

with col3:
    totChol = st.number_input("Total Cholesterol (mg/dl)", 100, 600, 200)
    sysBP = st.number_input("Systolic BP (mm Hg)", 80, 250, 120)
    diaBP = st.number_input("Diastolic BP (mm Hg)", 40, 150, 80)

with col4:
    BMI = st.number_input("BMI", 15.0, 60.0, 25.0, step=0.1)
    heartRate = st.number_input("Heart Rate (bpm)", 40, 200, 75)
    glucose = st.number_input("Glucose (mg/dl)", 50, 400, 80)

with col5:
    st.write("✨ Ready to predict?")
    st.write("")
    st.write("")
    predict_button = st.button("Predict 10-Year CHD Risk", type="primary")

# --- PREDICTION LOGIC ---
if predict_button:
    input_data = np.array([[male, age, education, currentSmoker, cigsPerDay,
                            BPMeds, prevalentStroke, prevalentHyp, diabetes,
                            totChol, sysBP, diaBP, BMI, heartRate, glucose]])
    
    try:
        prediction = model.predict(input_data)[0]
        
        # Try to get probability, else fallback to decision function
        try:
            prediction_proba = model.predict_proba(input_data)[0][1]
        except AttributeError:
            # For SVC without probability=True
            score = model.decision_function(input_data)[0]
            prediction_proba = 1 / (1 + np.exp(-score))  # sigmoid approximation

        st.divider()
        st.subheader("💖 Prediction Result")

        # Colorful display with metrics and progress bar
        if prediction == 1:
            st.error(f"⚠️ High Risk: Patient has a {prediction_proba*100:.2f}% risk of 10-year CHD.")
            st.warning("Please consult a medical professional for further evaluation.")
            st.progress(min(prediction_proba, 1.0))  # progress bar for risk
        else:
            st.success(f"✅ Low Risk: Patient has a {prediction_proba*100:.2f}% risk of 10-year CHD.")
            st.info("This is an estimate. Regular check-ups are still recommended.")
            st.progress(min(prediction_proba, 1.0))

        # Show a simple gauge using st.metric
        st.metric(label="Estimated Risk %", value=f"{prediction_proba*100:.2f}%")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
