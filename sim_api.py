from modules.stock_info_lstm import CandleLSTM
from data.read_aggregated_data import read_stock_values
import fastapi

LSTM_CHECKPOINT_PATH = "/teamspace/studios/this_studio/TraderAgent/checkpoints/lstm-epoch=29-val_loss=0.1344.ckpt"

def load_candle_model(checkpoint_filepath, device='cpu'):
    """
    Load a trained CandleLSTM model from checkpoint.
    
    Args:
        checkpoint_filepath: Path to the model checkpoint file
        device: Device to load the model onto ('cpu' or 'cuda')
    
    Returns:
        Loaded CandleLSTM model
    """
    model = CandleLSTM.load_from_checkpoint(checkpoint_filepath)
    model = model.to(device)
    model.eval()
    return model

app = fastapi.FastAPI()
model = None
step = 0

@app.on_event("startup")
def startup_event():
    global model
    model = load_candle_model(LSTM_CHECKPOINT_PATH)
    print("CandleLSTM model loaded successfully.")

@app.get("/valuepredict")
def value_predict():
    if model is None:
        return {"error": "Model not loaded yet."}

    global step
    stock_values = read_stock_values(step)
    predictions = {}
    current = {}

    for ticker, bars in stock_values.items():
        if bars is None or len(bars) < 12:
            predictions[ticker] = None
            continue

        # Prepare input tensor
        import torch
        import numpy as np

        input_data = np.array([[bar['Open'], bar['High'], bar['Low'], bar['Close'], bar['Volume']] for bar in bars])

        # Normalize input
        base_values = input_data[0]
        normalized = input_data.copy()
        normalized[:, :4] = (normalized[:, :4] - base_values[:4]) / (base_values[:4] + 1e-8) * 100
        normalized[:, 4] = np.log1p(normalized[:, 4]) - np.log1p(base_values[4])

        input_tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)  # Shape: (1, seq_len, features)
        input_tensor = input_tensor.to(next(model.parameters()).device)

        # Make prediction
        with torch.no_grad():
            pred_tensor = model(input_tensor)

        # Denormalize prediction
        pred_array = pred_tensor.cpu().numpy()[0]
        denorm_pred = pred_array.copy()
        denorm_pred[:4] = (denorm_pred[:4] / 100) * (base_values[:4] + 1e-8) + base_values[:4]
        denorm_pred[4] = np.expm1(denorm_pred[4] + np.log1p(base_values[4]))

        #print("All good so far")

        predictions[ticker] = {
            'Open': float(denorm_pred[0]),
            'High': float(denorm_pred[1]),
            'Low': float(denorm_pred[2]),
            'Close': float(denorm_pred[3]),
            'Volume': float(denorm_pred[4])
        }

    current = {ticker: bars[-1] for ticker, bars in stock_values.items() if bars is not None and len(bars) > 0}
    print(current)
    # current = {ticker: bars[-1] for ticker, bars in stock_values.items()}
    
    step += 1

    result = {"predictions": predictions, "current": current}

    return result