import os

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from load_config import config_data

# Load config
path = config_data["data_path"]
csv_path = os.path.join(path, "data", "raw", "WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Import data
data = pd.read_csv(csv_path)
print(f"Shape of data before processing: {data.shape}")

# Encode categorical data
encode_columns = {"Yes": 1, "No": 0}
columns_to_encode = ["Attrition", "OverTime"]

for col in columns_to_encode:
    if col in data.columns:
        data[col] = data[col].map(encode_columns)

data.drop(columns=["EmployeeNumber", "EmployeeCount", "Over18"], inplace=True)
data.reset_index(drop=True, inplace=True)

# Features and target
X = data.drop(columns=["Attrition"])
y = data["Attrition"]

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
    remainder="passthrough",
)

# Pipeline = preprocessing + model
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", LGBMClassifier())])

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
pipeline.fit(X_train, y_train)

# Save model + template row
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(pipeline, os.path.join(MODEL_DIR, "pipeline.pkl"))
joblib.dump(X.iloc[0], os.path.join(MODEL_DIR, "template_features.pkl"))

print("✅ Training complete. Model and template saved.")
