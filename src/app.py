import os

import joblib
import pandas as pd
import streamlit as st

# Load pipeline and template
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
pipeline = joblib.load(os.path.join(MODEL_DIR, "pipeline.pkl"))
template = joblib.load(os.path.join(MODEL_DIR, "template_features.pkl"))

st.title("Employee Attrition Prediction")

# Minimal relevant inputs
age = st.number_input("Age", min_value=18, max_value=70, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
department = st.selectbox(
    "Department", ["Sales", "Research & Development", "Human Resources"]
)
job_role = st.selectbox(
    "JobRole",
    [
        "Sales Executive",
        "Research Scientist",
        "Laboratory Technician",
        "Manufacturing Director",
        "Healthcare Representative",
        "Manager",
        "Sales Representative",
        "Research Director",
        "Human Resources",
    ],
)
overtime = st.selectbox("OverTime", ["Yes", "No"])
monthly_income = st.number_input(
    "MonthlyIncome", min_value=1000, max_value=30000, value=5000
)
years_at_company = st.number_input("YearsAtCompany", min_value=0, max_value=40, value=3)

if st.button("Predict"):
    # Copy template and override editable fields
    input_data = template.copy()
    input_data["Age"] = age
    input_data["Gender"] = gender
    input_data["Department"] = department
    input_data["JobRole"] = job_role
    input_data["OverTime"] = overtime
    input_data["MonthlyIncome"] = monthly_income
    input_data["YearsAtCompany"] = years_at_company

    # Convert to DF before predicting
    input_df = pd.DataFrame([input_data])

    pred = pipeline.predict(input_df)[0]
    prob = pipeline.predict_proba(input_df)[0][1]

    st.write("Attrition:", "Yes" if pred == 1 else "No")
    st.write(f"Probability of attrition: {prob:.3f}")
