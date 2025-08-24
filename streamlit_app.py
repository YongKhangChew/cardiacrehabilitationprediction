import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load your trained model
#model = load_model("nn_model.h5")
model = tf.keras.models.load_model("nn_model.h5")

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
#risk_level_reverse = {v: k for k, v in encoders["Risk Level"].items()}
risk_level_reverse = {
    0: "high",
    1: "low",
    2: "moderate",
    3: "unknown"
}

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
    # Convert input_df to numpy with correct shape
    user_input = input_df.to_numpy()

    # Get raw model probabilities
    probs = model.predict(user_input, verbose=0)

    # Take the class with max probability
    pred_class = int(np.argmax(probs, axis=1)[0])

    # Map back to label
    risk_label = risk_level_reverse[pred_class]

    # Display outputs
    st.write("### 🔍 Model Outputs")
    st.write("Raw probabilities:", probs.tolist())
    st.write("Predicted class index:", pred_class)
    st.write("Final Risk Level Prediction:", f"**{risk_label.upper()}**")
