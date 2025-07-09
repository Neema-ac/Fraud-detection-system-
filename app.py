
import streamlit as st
import numpy as np
import joblib

model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Fraud Detection", page_icon="💳")
st.title("💳 Fraud Detection System")
st.markdown("Enter transaction data to predict if it's fraudulent.")
amount = st.number_input("Amount", value=0.0, step=0.01)
inputs = [amount]  # Start with the extra feature
for i in range(1, 29):  # V1 to V28
    val = st.number_input(f"V{i}", value=0.0, step=0.01)
    inputs.append(val)

if st.button("Predict Fraud"):
    input_array = np.array([inputs])
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")
