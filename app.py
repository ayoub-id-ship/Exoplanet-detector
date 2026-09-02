import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

st.set_page_config(page_title="Exoplanet Detector 1D-CNN", layout="wide")

st.title("🪐 AI Exoplanet Detection System")
st.write("1D Convolutional Neural Network trained on NASA Kepler Light Curves.")

@st.cache_resource
def load_trained_model():
    model = Sequential([
        Conv1D(filters=16, kernel_size=9, activation='relu', input_shape=(3197, 1)),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=7, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.load_weights("exoplanet_kepler_cnn.keras")
    return model

try:
    model = load_trained_model()
except Exception as e:
    st.error(f"Error loading model weights: {e}")
    st.stop()

st.sidebar.header("Light Curve Input")
sample_type = st.sidebar.selectbox("Choose a test signal profile:", ["Synthetic Transit (Exoplanet)", "Flat Star Signal (Non-Exoplanet)"])

time = np.linspace(0, 10, 3197)
if sample_type == "Synthetic Transit (Exoplanet)":
    flux = 1.0 + np.random.normal(0, 0.02, 3197)
    for center in [600, 1600, 2600]:
        flux[center - 15 : center + 15] -= 0.08
else:
    flux = 1.0 + np.random.normal(0, 0.02, 3197)

flux_norm = (flux - np.mean(flux)) / np.std(flux)
input_data = flux_norm[np.newaxis, ..., np.newaxis]

prob = float(model.predict(input_data)[0][0])

col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(time, flux, color="#1f77b4", linewidth=0.8)
    ax.set_ylabel("Normalized Flux")
    ax.set_xlabel("Time Point")
    st.pyplot(fig)

with col2:
    st.metric("Exoplanet Probability", f"{prob*100:.2f}%")
    if prob > 0.5:
        st.error("🚨 Exoplanet Transit Detected!")
    else:
        st.success("✅ No Exoplanet Detected")
