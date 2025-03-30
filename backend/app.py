from fastapi import FastAPI, UploadFile, File
import pandas as pd
import torch
import numpy as np
from io import StringIO
from fastapi.responses import JSONResponse
from sklearn.preprocessing import MinMaxScaler
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
# Load PyTorch model
MODEL_PATH = "/Users/ayushkushwaha/Desktop/Sem-8-final-project/backend/stock_transformer_model_stacked_volume.pt"

try:
    model = torch.load(MODEL_PATH, map_location=torch.device('cpu'),weights_only=False)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Prevent crashes if the model fails to load



def preprocess_data(data):
    """Normalize 'close' and 'volume', then return stacked sequence and last 8 close values."""
    if len(data) < 8:
        raise ValueError("Not enough data points. At least 8 rows are required.")
    
    data = data[['close', 'volume']].apply(pd.to_numeric, errors='coerce').fillna(0)

    scaler_close = MinMaxScaler()
    scaler_volume = MinMaxScaler()

    data['close'] = scaler_close.fit_transform(data[['close']])
    data['volume'] = scaler_volume.fit_transform(data[['volume']])

    # Stack close and volume in an alternating pattern
    stacked_seq = np.column_stack((data['close'].values[-8:], data['volume'].values[-8:])).flatten()

    # Last 8 normalized close values
    last_8_close = data['close'].values[-8:].astype('float32')

    return stacked_seq.astype('float32'), last_8_close


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read CSV file
    data = pd.read_csv(StringIO((await file.read()).decode("utf-8")))

    if not {'close', 'volume'}.issubset(data.columns):
        return JSONResponse(status_code=400, content={"error": "CSV must contain 'close' and 'volume' columns"})

    # Preprocess and create tensor input (1, 16)
    input_seq, last_8_close = preprocess_data(data)
    
    input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)

    # Predict the next value
    with torch.no_grad():
        next_value = model(input_tensor).item()

    return JSONResponse({"combined_values": last_8_close.tolist() + [next_value]})
