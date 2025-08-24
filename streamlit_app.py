import streamlit as st
import pandas as pd
import joblib

# Load your trained model (replace with your actual file)
# model = joblib.load("model.pkl")

# Encoders mapping dictionary (from your LabelEncoder output)
encoders = {
    "ROM": {"abnormal": 0, "normal": 1, "unknown": 2},
    "Balance in Sitting and Standing": {"no": 0, "unknown": 1, "yes": 2},
    "Functional Activity": {"assisted": 0, "independent": 1, "unknown": 2},
    "Walking": {"dependent": 0, "independent": 1, "unknown": 2},
    "Gait": {"abnormal": 0, "normal": 1, "unknown": 2},
    "Posture": {"abnormal": 0, "normal": 1, "unknown": 2},
    "Risk Level": {"high": 0, "low": 1, "moderate": 2, "unknown": 3},
    "RISK  - Risk Type": {
        "high": 0,
        "low": 1,
        "low to moderate": 2,
        "moderate": 3,
        "moderate to high": 4,
        "unknown": 5,
    },
}

# Reverse mapping for Risk Level (to display prediction nicely)
risk_level_reverse = {v: k for k, v in encoders["Risk Level"].items()}

st.title("Cardiac Rehab Risk Prediction")

st.write(
    """
    This app predicts the **Risk Level** of a patient based on their 
    health assessment features. Each categorical input is encoded numerically
    — the mappings are shown below for clarity.
    """
)

# Sidebar to show the mappings
st.sidebar.header("Feature Bindings (Encodings)")
for feature, mapping in encoders.items():
    st.sidebar.subheader(feature)
    for k, v in mapping.items():
        st.sidebar.write(f"{k} → {v}")

# Collect user inputs
st.header("Enter Patient Information")

user_data = {}
for feature, mapping in encoders.items():
    if feature == "Risk Level":  # this is the output, skip input
        continue
    choice = st.selectbox(f"{feature}", list(mapping.keys()))
    user_data[feature] = mapping[choice]

# Convert input to DataFrame
input_df = pd.DataFrame([user_data])

# Prediction button
if st.button("Predict Risk Level"):
    pred = model.predict(input_df)[0]
    risk_label = risk_level_reverse[pred]

    st.success(f"**Predicted Risk Level:** {risk_label} ({pred})")
