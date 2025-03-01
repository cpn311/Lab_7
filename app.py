import pandas as pd
import numpy as np
import pickle
import streamlit as st
import matplotlib.pyplot as plt
# Title of the app
st.title("My Streamlit App")
st.write("Welcome to my Streamlit app!")

# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
file_path = "AmesHousing.xlsx"  # Update this with your file path
df = pd.read_excel(file_path, sheet_name="AmesHousing")

# Display the first few rows of the dataset
st.write("Dataset Preview:")
st.dataframe(df.head())  # Display the first few rows of the dataset

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

# Display the evaluation results
st.write(f"Mean Absolute Error: {mae}")
st.write(f"Root Mean Squared Error: {rmse}")

# Optionally, you can show a feature importance plot
import matplotlib.pyplot as plt

# Plot feature importances
feature_importances = model.feature_importances_
sorted_idx = np.argsort(feature_importances)[::-1]
top_features = X.columns[sorted_idx][:10]  # Display top 10 features
top_importances = feature_importances[sorted_idx][:10]

fig, ax = plt.subplots()
ax.barh(top_features, top_importances)
ax.set_xlabel("Feature Importance")
ax.set_title("Top 10 Feature Importances")
st.pyplot(fig)

