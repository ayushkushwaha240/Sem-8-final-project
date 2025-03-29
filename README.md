# Sem-8 Final Project  

## Project Overview  
This is my final year project. This project aims to incorporate the concept of volume divergence into a model to predict the next high or low stock price.  

## Understanding Volume Divergence  
Volume divergence is a simple yet effective way to analyze investor interest. It helps us understand how **prices change** in response to shifts in **average volume**.  

### What is Volume?  
Volume is a measure of stock transactions. If **1 share is bought and sold**, the volume count is **1**. If **1M shares are transacted**, the volume is **1M**.  

### Types of Volume Divergence:  
1. **Positive Confirmed Divergence**  
2. **Negative Confirmed Divergence**  
3. **Negative Fake Divergence**  
4. **Positive Fake Divergence**  

## Dataset Preparation  
We will use **TradingView API** or **NSE dataset** for stock market data.  
- **Time Frame:** 30-minute intervals  
- **Data Sources:** Top 20 companies from **Nifty50 (India)**  
- **Inputs:** Volume candlestick patterns, support, and resistance  
- **Output:** Prediction of the **next 3 candlesticks**  

### Candlestick Components:  
- **Opening Price**  
- **Highest Price**  
- **Lowest Price**  
- **Closing Price**  
- **Volume**  

## Incorporating Volume Information  
To analyze the role of volume in price movements, we experimented with three different approaches:  

### 1. **Without Volume Information**  
- The model was trained only on **closing prices** without considering the volume.  

### 2. **Volume Stacking Method**  
- The input sequence alternates between **close price** and **volume**:  
  ```plaintext
  close(t), volume(t), close(t+1), volume(t+1), ...
- Result: After evaluating the models, the volume stacking method achieved the lowest loss, making it the chosen approach for better predictions.
### 3. **Volume Addition Method (Final Approach)**
- Instead of treating volume separately, we normalized both close prices and volume and added them together:
- input = normalized_close + normalized_volume

## Tools Used
- Python
- Pandas, NumPy, Scikit-learn, PyTorch, MLflow, Streamlit, Docker, FastAPI
- TradingView API and NSE dataset

## Deployment
The frontend is deployed on Streamlit and can be accessed here:
https://sem-8-final-project-83uanlndy4ehjvnn8nlrrn.streamlit.app/

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/Sem-8-Final-Project.git
   cd Sem-8-Final-Project
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
3. **Run the backend:**
   ```bash
   uvicorn app:main --reload
4. **To run the frontend use:**
   ```bash
   streamlit run frontend.py

**Note:**
- Make sure about the directories in which the deployable files are
- Also, the model which is used for deployment purposes is the model in which I used the volume addition approach.

