import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

BACKEND_URL = "https://ayushkush2402-sem-8-project.hf.space/predict/"  # Change if backend is hosted elsewhere
# BACKEND_URL = "http://localhost:8000/predict/"

st.title("Stock Trend using Volume Divergence")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)
    
    # Send file to backend
    files = {"file": uploaded_file.getvalue()}
    response = requests.post(BACKEND_URL, files=files)
    
    if response.status_code == 200:
        result = response.json()
        combined_values = result["combined_values"]  # Expecting 9 values
        
        if len(combined_values) == 9:
            fig, ax = plt.subplots()

            # Original X values (time steps)
            x_values = np.arange(1, 10)  # Start indexing from 1

            # Interpolation for smooth curve
            x_smooth = np.linspace(x_values.min(), x_values.max(), 150)
            spl = make_interp_spline(x_values, combined_values, k=3)
            y_smooth = spl(x_smooth)

            # Shade the last value differently (Predicted Trend)
            ax.axvspan(8.5, 9.5, color='#FF5733', alpha=0.5)  # Highlight last value in orange

            # Plot the curve
            ax.plot(x_smooth, y_smooth, color='blue', linewidth=2, label='Stock Values')

            # Labels and legend
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Normalized Values")
            ax.set_title("Stock Prediction (Highlighting Last Value)")
            ax.legend()

            # Add label below the graph for predicted trend
            st.pyplot(fig)
            st.markdown("### **The red region shows the predicted trend**", unsafe_allow_html=True)
        else:
            st.error(f"Expected 9 values, but received {len(combined_values)}")
    else:
        st.error("Error in fetching predictions from backend")

st.warning("⚠️ **Disclaimer:** This is a project for educational purposes only. Do not use it for real trading or investment decisions.")