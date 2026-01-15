from modules.stock_info_lstm import CandleLSTM
from modules.news_info_sllm import StockLLMAgent
from data.read_aggregated_data import read_stock_values, read_stock_values_v2, read_stock_news_v2
import fastapi

LSTM_CHECKPOINT_PATH = "/teamspace/studios/this_studio/TraderAgent/checkpoints_lstm_1d_new/lstm-epoch=41-val_loss=5.8547.ckpt"

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

def load_news_model():
    llm_agent = StockLLMAgent(
        model_name="llama3.2:3b",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        prompt_yaml_path="/teamspace/studios/this_studio/TraderAgent/config/config_class.yaml",
        temperature=0.0,
    )
    print("Stock LLM agent loaded successfully.")
    return llm_agent

app = fastapi.FastAPI()
model = None
step = 0


@app.on_event("startup")
def startup_event():
    global model
    model = load_candle_model(LSTM_CHECKPOINT_PATH)
    print("CandleLSTM model loaded successfully.")

    global llm_agent
    llm_agent = load_news_model()
    print("Classification model loaded successfully.")

    global dates_list
    with open("/teamspace/studios/this_studio/TraderAgent/data/dates.txt", "r") as f:
        dates_list = [line.strip() for line in f if line.strip()]


@app.get("/valuepredict")
def value_predict():
    if model is None:
        return {"error": "Model not loaded yet."}

    global step
    #NOTE: step - inseamna a cata zi ii - adica indexul pentru datele din stock_values.json
    #NOTE: este un fisier dates.txt si de acolo se ia ziua curenta - ACTUALLY FUCK IT...


    # de aici luam date-ul curent...- dar sub forma unui index 
    with open("/teamspace/studios/this_studio/TraderAgent/data/current_date.txt", "r") as f:
        step = int(f.read().strip())

    print(f"Extracting stocks for {dates_list[step]}...")

    stock_values = read_stock_values_v2(step - 12)       # Pentru ca step e cea curenta si pe noi ne intereseaza cu 12 in urma...
    predictions = {}
    current = {}

    for ticker, bars in stock_values.items():
        if bars is None or len(bars) < 12:
            predictions[ticker] = None
            continue

        # Prepare input tensor
        import torch
        import numpy as np

        input_data = np.array([[bar['Open'], bar['High'], bar['Low'], bar['Close']] for bar in bars])

        # Normalize input
        base_values = input_data[0]
        normalized = input_data.copy()
        normalized[:, :4] = (normalized[:, :4] - base_values[:4]) / (base_values[:4] + 1e-8) * 100
        # normalized[:, 4] = np.log1p(normalized[:, 4]) - np.log1p(base_values[4])

        input_tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)  # Shape: (1, seq_len, features)
        input_tensor = input_tensor.to(next(model.parameters()).device)

        # Make prediction
        with torch.no_grad():
            pred_tensor = model(input_tensor)

        # Denormalize prediction
        pred_array = pred_tensor.cpu().numpy()[0]
        denorm_pred = pred_array.copy()
        denorm_pred[:4] = (denorm_pred[:4] / 100) * (base_values[:4] + 1e-8) + base_values[:4]
        # denorm_pred[4] = np.expm1(denorm_pred[4] + np.log1p(base_values[4]))

        #print("All good so far")

        predictions[ticker] = {
            'Open': float(denorm_pred[0]),
            'High': float(denorm_pred[1]),
            'Low': float(denorm_pred[2]),
            'Close': float(denorm_pred[3]),
            # 'Volume': float(denorm_pred[4])
        }

    current = {ticker: bars[-1] for ticker, bars in stock_values.items() if bars is not None and len(bars) > 0}
    # print(current)
    # current = {ticker: bars[-1] for ticker, bars in stock_values.items()}
    
    # step += 1

    result = {"predictions": predictions, "current": current}

    return result


@app.get("/analyzenews")
def analyze_news():
    """
    Endpoint to classify or analyze stock news via LLM.
    
    Args:
        news: Stock news text
    
    Returns:
        LLM response text
    """
    if llm_agent is None:
        return {"error": "LLM agent not loaded yet."}

    with open("/teamspace/studios/this_studio/TraderAgent/data/current_date.txt", "r") as f:
        step = int(f.read().strip())

    global dates_list
    print(dates_list[step])

    stock_news = read_stock_news_v2(dates_list[step], max_items=3)

    all_responses = {}

    for ticker, news in stock_news.items():
        final_string = ""
        for n in news:
            # final_string += n["headline"] + "---"
            final_string += n["summary"] + ";\n"
        # return {"response": final_string}
        response = llm_agent.run(final_string)
        all_responses[ticker] = response
    # print(stock_news)
    return {"response": all_responses}

    #TODO: poate putin promptul de rafinat oleaca...
    # Exemplu de return:
    # {"response":{"AAPL":"Category: Positive","NVDA":"Category: Positive","AMZN":"Negative","GOOGL":"Category: Positive",
    # "MSFT":"Category: Positive","TSLA":"Category: Neutral","META":"Category: Positive","NFLX":"Category: Positive",
    # "INTC":"Neutral","ORCL":"Category: Neutral","PLTR":"Category: Neutral"}}

