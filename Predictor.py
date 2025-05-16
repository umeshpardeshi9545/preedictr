# -*- coding: utf-8 -*-
"""
Created on Tue May 13 12:00:20 2025

@author: SANKET NICHAT
"""

import streamlit as st
import pickle
import pandas as pd
import urllib.request
from sklearn.preprocessing import LabelEncoder
# Re-create and fit LabelEncoder (Ensure labels are the same as during training)
label_mapping = ['Depression', 'Diabetes, Type 2', 'High Blood Pressure']  # Add all your conditions
le = LabelEncoder()
le.fit(label_mapping)

# Load the trained model
url = "https://raw.githubusercontent.com/umeshpardeshi9545/preedictr/main/logistic_regression_model.pkl"

# Load the model
try:
    with urllib.request.urlopen(url) as response:
        model_data = response.read()  # Read bytes
        model = pickle.loads(model_data)  # Load model from bytes
except Exception as e:
    model = None
    st.error(f"Error loading the model: {e}")
# Load the TF-IDF vectorizer
url1 = "https://raw.githubusercontent.com/umeshpardeshi9545/preedictr/main/tfidf_vectorizer.pkl"

# Load the model
try:
    with urllib.request.urlopen(url1) as response:
        model_data1 = response.read()  # Read bytes
        vectorizer = pickle.loads(model_data1)  # Load model from bytes
except Exception as e:
    model = None
    st.error(f"Error loading the model: {e}")

# Streamlit UI
st.title("Medical Condition Prediction from Reviews")
st.write("Enter a medical review to predict its associated condition.")

# User input
user_input = st.text_area("Review:")

if st.button("Predict Condition"):
    if user_input:
        transformed_input = vectorizer.transform([user_input])  # Vectorize input
        prediction = model.predict(transformed_input)  # Make prediction
        
        # Convert predicted label back to text format
        predicted_condition = le.inverse_transform([prediction[0]])  # Convert number to condition name

        # Display result in Streamlit
        st.success(f"Predicted Condition: **{predicted_condition[0]}**")

