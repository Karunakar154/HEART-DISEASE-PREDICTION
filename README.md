🫀 Heart Disease Prediction App
🔍 Overview
This is an interactive Streamlit web app that predicts the 10-year risk of Coronary Heart Disease (CHD) based on medical and demographic data.
It uses a machine learning model (SVC / Logistic Regression / Random Forest, etc.) trained on heart disease risk factors.
# 🫀 Heart Disease Prediction App
[https://karunakar154-heart-disease-prediction-app-p9h6kc.streamlit.app]
🌟 Features

✅ Predicts 10-year CHD risk using 15 medical and lifestyle inputs
✅ Interactive, user-friendly UI built with Streamlit
✅ Shows color-coded risk levels (Low / High)
✅ Includes progress bar and risk percentage metric
✅ Automatically handles models with or without predict_proba()
🧠 Machine Learning Model

The model (model.pkl) is trained using features such as:

Sex

Age

Education level

Smoking habits (current smoker, cigarettes/day)

Blood pressure medication

Stroke & hypertension history

Diabetes
🧑‍💻 Tech Stack

Python

Streamlit

scikit-learn

NumPy, Pandas

Pickle (for model serialization)
📊 Output Example

Low Risk Example:

✅ “Patient has a 7.34% risk of 10-year CHD. Regular check-ups recommended.”

High Risk Example:

⚠️ “Patient has a 79.24% risk of 10-year CHD. Consult a cardiologist immediately.”
🚀 Future Improvements

Add data visualization for patient comparison

Connect to a real-time health database

Deploy on Streamlit Cloud or Hugging Face Spaces

Cholesterol, BP, BMI, heart rate, glucose

You can retrain your model with probability=True (for SVC) to get true probabilities.
