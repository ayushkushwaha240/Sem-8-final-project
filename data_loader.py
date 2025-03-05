import yfinance as yf
import pandas as pd
import os

# Load the CSV file (assuming the file has a 'Ticker' column)
df = pd.read_csv('/Users/ayushkushwaha/Desktop/Sem-8-final-project/Nasdaq 100 (NDX).csv')

# Take the top 25 companies
top_25_tickers = df['Name'].head(25).tolist()

# Create output folder if it doesn't exist
os.makedirs('nasdaq100_top25_charts', exist_ok=True)

# Download data for each of the top 25 companies
for ticker in top_25_tickers:
    print(f"Downloading data for {ticker}...")

    try:
        stock_data = yf.download(ticker, interval='30m', period='60d')

        if stock_data.empty:
            print(f"No data found for {ticker}, skipping.")
            continue

        # Save data to CSV
        stock_data.to_csv(f'nasdaq100_top25_charts/{ticker}_30min_60days.csv')

    except Exception as e:
        print(f"Failed to download data for {ticker}: {e}")

print("Data download complete for top 25 companies.")