# Classification Project

End‑to‑end binary (or multi‑class) classification mini project.

Overview

- Load data, explore class balance and missing values
- Clean, encode categorical features, scale numeric, stratified splits
- Train baseline (Dummy / Logistic Regression) then improved models (RandomForest, Gradient Boosting, etc.)
- Evaluate with accuracy, precision, recall, F1, ROC AUC, confusion matrix, ROC curve
- Select best model via cross‑validation + simple hyperparameter tuning
- Save final model and key metric/plot artifacts

Structure
data/ raw or processed data
notebooks/ quick EDA and trials
src/ training, preprocessing, inference code
models/ saved model (pkl)
reports/ metrics and figures

Quick Start

1. python -m venv .venv && source .venv/Scripts/activate (Windows adjust accordingly)
2. pip install -r requirements.txt
3. Place raw dataset into data/
4. Run notebooks/EDA.ipynb (optional)
5. python src/train.py
6. python src/evaluate.py

Use Model
from src.inference import load_model, predict
model = load_model("models/best_model.pkl")
preds = predict(model, new_dataframe)

Next (optional)

- Add calibration, SHAP, experiment tracking, simple API / UI
  Update placeholders (dataset source, target name, metrics) when available.
