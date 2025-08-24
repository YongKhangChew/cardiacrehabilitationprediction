import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load trained model
model = load_model("nn_model.h5")

# If you saved features.pkl earlier, load it
try:
    features = joblib.load("features.pkl")
except:
    features = [f"Feature_{i}" for i in range(7)]  # fallback

st.title("🔮 Risk Level Prediction (Neural Network)")

st.write("Enter the values for each feature to predict risk level:")

# Collect user input
user_input = {}
for col in features:
    user_input[col] = st.number_input(f"{col}", value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

if st.button("Predict"):
    # Predict with NN
    prediction = model.predict(input_df)
    pred_class = np.argmax(prediction, axis=1)[0]

    st.success(f"✅ Predicted Risk Level: **{pred_class}**")
