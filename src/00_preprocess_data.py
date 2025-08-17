import pandas as pd
import sys
import numpy as np

# Import data 
data = pd.read_csv("C:/Users/user/Desktop/MiniProject Data Scientist/Classification Project/Data//raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")
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