import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

BACKEND_URL = "http://127.0.0.1:8000/predict/"  # Change if backend is hosted elsewhere

st.title("Stock Prediction with Volume Divergence")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)
    
    # Send file to backend
    files = {"file": uploaded_file.getvalue()}
    response = requests.post(BACKEND_URL, files=files)
    
    if response.status_code == 200:
        result = response.json()
        combined_values = result["combined_values"]  # Expecting 11 values
        
        if len(combined_values) == 11:
            fig, ax = plt.subplots()

            # Original X values (time steps)
            x_values = np.arange(11)

            # Stricter interpolation
            x_smooth = np.linspace(x_values.min(), x_values.max(), 150)  # Reduced points for sharper curve
            spl = make_interp_spline(x_values, combined_values, k=3)  # Lower degree for strictness
            y_smooth = spl(x_smooth)

            # Shade the last 3 values' background with darker red
            ax.axvspan(8, 10, color='#FF9999', alpha=0.4)  # Darker red background

            # Add label inside the red background
            ax.text(8.5, max(combined_values) * 0.9, "Predicted Trend", 
                    fontsize=12, color='black', fontweight='bold')

            # Plot strict curve
            ax.plot(x_smooth, y_smooth, color='blue', linewidth=2, label='Stock Values')

            # Labels and legend
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Normalized Values")
            ax.set_title("Stock Prediction (Stricter Curve)")
            ax.legend()

            st.pyplot(fig)
        else:
            st.error(f"Expected 11 values, but received {len(combined_values)}")
    else:
        st.error("Error in fetching predictions from backend")
