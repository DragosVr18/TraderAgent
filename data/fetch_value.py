import yfinance as yf

def fetch_stock_value(ticker_symbol, interval, period, count=None):
    """
    Fetch the latest stock bars for a given ticker symbol.

    Parameters:
    ticker_symbol (str): Stock ticker symbol.
    count (int): Number of latest data points to fetch.
    interval (str): Data interval (e.g., '1m', '5m', '15m', '1h', '1d').
    period (str): Data period (e.g., '1d', '5d', '1mo', '3mo').

    Returns:
    DataFrame: A DataFrame containing the latest stock data.
    """
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(interval=interval, period=period)
    
    if not hist.empty:
        if count is None or count > len(hist):
            count = len(hist)
        latest_data = hist.tail(count)
        return latest_data
    else:
        raise ValueError(f"No data found for ticker symbol: {ticker_symbol}")
    

if __name__ == "__main__":
    symbol = "AAPL"
    count = None
    stock_data = fetch_stock_value(symbol, count=count, interval='5m', period='60d')
    print(len(stock_data))