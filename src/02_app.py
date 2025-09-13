import streamlit as st
import joblib
import numpy as np 
import pandas as pd
from load_config import load_config , config_data

config = load_config()
path = config["data_path"]

st.title("Predict Attrtition based on Job Experience")
st.write("Fill in the information below to predict attrition")

@st.cache_resource
def load_pipeline():
    return joblib.load(f"{path}src/pipeline_model.pkl")

pipeline = load_pipeline()

# Collect input from user
overtime = st.selectbox("Overtime" , ['Yes','No'])
jobrole = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician",
    "Manager", "Human Resources", "Sales Representative"])
stocklevel = st.selectbox("Stock Option Level", [0,1])
age = st.slider("Age", 18,60,30)

feature = {
    "Age": age,
    "JobRole": jobrole,
    "StockOptionLevel": stocklevel,
    "Overtime": overtime
}

input_data = pd.DataFrame([feature])

# Prediction button 
if st.button("Predict attrition"):
    
    prob = pipeline.predict_proba(input_data)[0][1]
    prediction = pipeline.predict(input_data)[0]

    st.subheader("Prediction Result")
    st.write("Employee likely to leave" if prediction == 1 else "Employee likely to stay")
    st.write(f"Probabilty of attrition {prob:.2f%}")
    
"""
# Explainability button 
if st.button("Explain Prediction"):
    
    shap_values = explainer(input_data)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.plots.waterfall(shap_values[0])
    st.pyplot(bbox_inches='tight')
"""





