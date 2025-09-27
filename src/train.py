import pandas as pd
import joblib
import numpy as np 
import os
from load_config import load_config
from lightgbm import LGBMClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

config = load_config()
path = config["data_path"]

# === Load Data ===
data = pd.read_csv(f"{path}/Data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")
print(f"Shape of data before merge: {data.shape}")

# === Encode Binary Columns ===
encode_columns = {"Yes": 1, "No": 0}
for col in ["Attrition", "OverTime"]:
    data[col] = data[col].map(encode_columns)

data.drop(columns=["EmployeeNumber", "EmployeeCount", "Over18"], inplace=True)
data.reset_index(drop=True, inplace=True)

# === Split ===
X = data.drop(columns=["Attrition"])
y = data["Attrition"]

cat_cols = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown="ignore"), cat_cols)],
    remainder="passthrough"
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LGBMClassifier())
])

# === Train ===
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
pipeline.fit(X_train, y_train)

# === Save Model ===
models_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(models_dir, exist_ok=True)

MODEL_PATH = os.path.join(models_dir, "pipeline.pkl")
joblib.dump(pipeline, MODEL_PATH)

print(f"✅ Saved model to: {MODEL_PATH}")
print("Training complete.")
