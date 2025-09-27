import streamlit as st
import joblib
import numpy as np 
import pandas as pd
import os
from load_config import load_config

config = load_config()
path = config["data_path"]

# === Load Model ===
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "pipeline.pkl")
pipeline = joblib.load(MODEL_PATH)

st.title("Employee Attrition Prediction")

# === Collect Inputs ===
age = st.number_input("Age", min_value=18, max_value=70, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
job_role = st.selectbox("JobRole", [
    "Sales Executive", "Research Scientist", "Laboratory Technician",
    "Manufacturing Director", "Healthcare Representative",
    "Manager", "Sales Representative", "Research Director",
    "Human Resources"
])
overtime = st.selectbox("OverTime", ["Yes", "No"])
monthly_income = st.number_input("MonthlyIncome", min_value=1000, max_value=30000, value=5000)
years_at_company = st.number_input("YearsAtCompany", min_value=0, max_value=40, value=3)

# === Predict ===
if st.button("Predict"):
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Department": department,
        "JobRole": job_role,
        "OverTime": overtime,
        "MonthlyIncome": monthly_income,
        "YearsAtCompany": years_at_company,
    }])

    pred = pipeline.predict(input_df)[0]
    prob = pipeline.predict_proba(input_df)[0][1]

    st.write("Attrition:", "Yes" if pred == 1 else "No")
    st.write(f"Probability of attrition: {prob:.3f}")
