import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from tensorflow.keras.utils import Sequence

# Load Dataset (Replace with actual S&P Global data)
data = pd.read_csv("sp_global_stock_data.csv", parse_dates=['Date'], index_col='Date')
print(data.head())

# Handle Missing Values
data.fillna(method='ffill', inplace=True)

# Feature Engineering (Example: Moving Averages)
data['SMA_7'] = data['Close'].rolling(window=7).mean()
data['SMA_30'] = data['Close'].rolling(window=30).mean()

# Normalize Data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.drop(columns=['Volume']))
data_scaled = pd.DataFrame(data_scaled, columns=data.columns.drop('Volume'), index=data.index)

# Sequence Preparation
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i+seq_length].values)
        y.append(data.iloc[i+seq_length]['Close'])
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(data_scaled, seq_length)

# Convert to TensorFlow Tensors
X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)

# Create Dataset and DataLoader
class StockDataset(Sequence):
    def __init__(self, X, y, batch_size=32):
        self.X = X
        self.y = y
        self.batch_size = batch_size
    
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = start + self.batch_size
        return self.X[start:end], self.y[start:end]

train_dataset = StockDataset(X_tensor, y_tensor, batch_size=32)

print("Dataset Ready with", len(X_tensor), "samples")