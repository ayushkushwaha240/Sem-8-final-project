from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import torch
import joblib
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Load the trained model and encoder
model = torch.jit.load("stock_transformer_model.pt")  # Load your trained model
model.eval()  # Set model to evaluation mode
label_encoder = joblib.load("stock_label_encoder.pkl")

def preprocess_data(file):
    """Read CSV and preprocess it."""
    df = pd.read_csv(file)
    df["Stock"] = label_encoder.transform(df["Stock"])  # Encode stock names
    return df

class StockPredictionRequest(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    stock: str  # Stock name

@app.post("/upload/")
async def upload_stock_data(file: UploadFile = File(...)):
    try:
        df = preprocess_data(file.file)
        return {"message": "File processed successfully", "data_preview": df.head().to_dict()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/")
async def predict_stock(data: StockPredictionRequest):
    try:
        # Convert input data to tensor
        input_tensor = torch.tensor([
            [data.open, data.high, data.low, data.close, data.volume, label_encoder.transform([data.stock])[0]]
        ], dtype=torch.float32)
        
        input_tensor = input_tensor.to("cuda" if torch.cuda.is_available() else "cpu")
        model.to(input_tensor.device)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor).cpu().numpy().tolist()
        
        return {"stock": data.stock, "predicted_value": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/decode/{encoded_value}")
async def decode_stock(encoded_value: int):
    try:
        stock_name = label_encoder.inverse_transform([encoded_value])[0]
        return {"encoded_value": encoded_value, "stock_name": stock_name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
