# Sem-8 Final Project  

## Project Overview  
This is my final year project. I aim to develop a model based on volume divergence to predict the next high or low in stock prices.

## Understanding Volume Divergence  
Volume divergence is a simple yet effective way to analyze investor interest. It helps us understand how prices change in response to shifts in average volume.  

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

## Technologies Used  
- **Python**  
- **Pandas, NumPy, Scikit-learn, TensorFlow/PyTorch**  
- **TradingView API / NSE dataset**  

## Installation  
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/Sem-8-Final-Project.git
   cd Sem-8-Final-Project
