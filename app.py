import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
file_path = "AmesHousing.xlsx"  # Update this with your file path
df = pd.read_excel(file_path, sheet_name="AmesHousing")

# Drop non-essential columns
df = df.drop(columns=['Order', 'PID'])

# Handle missing values and encode categorical variables
for col in df.columns:
    if df[col].dtype == "object":  # Categorical column
        df[col].fillna(df[col].mode()[0], inplace=True)
        df[col] = LabelEncoder().fit_transform(df[col])  # Encode categorical data
    else:  # Numeric column
        df[col].fillna(df[col].median(), inplace=True)

# Split into features and target variable
X = df.drop(columns=['SalePrice'])
y = df['SalePrice']

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5  # Manually take the square root

print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")

