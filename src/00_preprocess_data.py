import pandas as pd
import sys
import numpy as np
from config import load_config

config = load_config()

path = config["data_path"]

# Import data 
data = pd.read_csv(path)
print(f"Shape of data before merge: {data.shape}")

# Encode categorical data 
encode_columns = {
    "Yes": 1,
    "No": 0
}

columns_to_encode = [
    "Attrition",
    "OverTime",
]

for column in data.columns:
    if column in columns_to_encode:
        data[column] = data[column].map(encode_columns)

data.drop(columns = ["EmployeeNumber","EmployeeCount","Over18"], inplace = True)

data.to_csv("C:\\Users\\user\\Desktop\\MiniProject Data Scientist\\Classification Project\\Data\\processed\\processed_data.csv",index = False)