import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# ---------------------------
# Load the trained model (ignore compile errors)
# ---------------------------
model = load_model("model.h5", compile=False)

# ---------------------------
# Streamlit app
# ---------------------------
st.title("🔮 Deep Learning Model Deployment")
st.write("write the values for predict: ")

# ---------------------------
# User Inputs
# ---------------------------
n_features = model.input_shape[1]

inputs = []
for i in range(n_features):
    value = st.number_input(f"Feature {i+1}", value=0.0)
    inputs.append(value)

# ---------------------------
# Predict Button
# ---------------------------
if st.button("Predict"):
    input_data = np.array([inputs])

    prediction = model.predict(input_data)

    # لو multi-class classification
    if prediction.shape[1] > 1:
        predicted_class = np.argmax(prediction, axis=1)[0]
        st.success(f"✅ Predicted Class: {predicted_class} (probabilities = {prediction[0]})")
    else:
        # Binary classification
        predicted_class = (prediction > 0.5).astype("int32")[0][0]
        st.success(f"✅ Predicted Class: {predicted_class} (probability = {prediction[0][0]:.4f})")
