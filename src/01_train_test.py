import pandas as pd
import joblib
import numpy as np 
from lightgbm import LGBMClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# Read processed data 
processed_data = pd.read_csv("C:/Users/user/Desktop/MiniProject Data Scientist/Classification Project/Data/processed_data.csv")
processed_data.reset_index(drop = True, inplace = True)

# Store columns
X = processed_data.drop(columns=["Attrition"])
y = processed_data["Attrition"]

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ],
    remainder="passthrough"
)

# Full pipeline = preprocessing + model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LGBMClassifier())
])

# Train
X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, random_state=42)
pipeline.fit(X_train, y_train)

# Save the model 
joblib.dump(pipeline,"C:\\Users\\user\\Desktop\\MiniProject Data Scientist\\Classification Project\\src\\pipeline_model.pkl")
