from fastapi import FastAPI, UploadFile, File
import pandas as pd
import torch
import io
from fastapi.responses import JSONResponse, FileResponse

app = FastAPI()

# Load PyTorch model
MODEL_PATH = "stock_transformer_model.pt"
model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.eval()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    df, input_tensor = preprocess_data(file)
    
    with torch.no_grad():
        prediction = model(input_tensor)
    
    pred_values = prediction.numpy().flatten()
    df['prediction'] = pred_values  # Append predictions to DataFrame
    
    # Save the DataFrame as CSV
    csv_filename = "predictions.csv"
    df.to_csv(csv_filename, index=False)
    
    return FileResponse(csv_filename, media_type="text/csv", filename=csv_filename)
