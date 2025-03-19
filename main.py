from fastapi import FastAPI, UploadFile, File
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from fastapi.responses import JSONResponse, FileResponse

app = FastAPI()

# Load PyTorch model
MODEL_PATH = "stock_transformer_model.pt"
model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.eval()

def preprocess_data(data):
    """Preprocess DataFrame."""
    scaler_close = MinMaxScaler()
    scaler_volume = MinMaxScaler()
    
    data['close'] = scaler_close.fit_transform(data[['close']])
    data['volume'] = scaler_volume.fit_transform(data[['volume']])
    
    return data, scaler_close, scaler_volume

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    data = pd.read_csv(file.file)
    data, scaler_close, scaler_volume = preprocess_data(data)
    
    # Select last 5 values as input
    input_seq = data[['close', 'volume']].values[-5:] 
    input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    
    predictions = []
    for _ in range(3):  # Predict next 3 steps iteratively
        with torch.no_grad():
            pred = model(input_tensor)
        
        pred_value = pred.numpy().flatten()[0]  # Extract single prediction
        predictions.append(pred_value)
        
        # Stack the predicted value and update input tensor
        new_input = torch.tensor([[pred_value, input_tensor[0, -1, 1].item()]], dtype=torch.float32)
        input_tensor = torch.cat((input_tensor[:, 1:, :], new_input.unsqueeze(0)), dim=1)  # Shift left and append new value
    
    # Convert predictions back to original scale
    predictions = scaler_close.inverse_transform([[p] for p in predictions]).flatten().tolist()
    
    return JSONResponse({"predictions": predictions})