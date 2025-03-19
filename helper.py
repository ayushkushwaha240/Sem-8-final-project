import pandas as pd
import torch

def preprocess_data(file: UploadFile):
    df = pd.read_csv(file.file)
    
    features = df[['open', 'high', 'low', 'close', 'volume']].values
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    return df, features

def make_sequence():
    pass