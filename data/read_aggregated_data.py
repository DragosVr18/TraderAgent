import pandas as pd
import numpy as np
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data_aggregated_v4'
NEWS_COUNT = 3
BAR_COUNT = 12

def read_stock_values(step):
    """
    Read the next BAR_COUNT bars for each ticker starting from the given step.
    
    Parameters:
    step (int): The starting index (0-based) for reading bars.
    
    Returns:
    dict: Dictionary with ticker symbols as keys and list of 4 bars as values.
          Returns None for tickers that don't have enough data.
    """
    with open(DATA_DIR / 'stock_values.json', 'r') as f:
        all_data = json.load(f)
    
    result = {}
    for ticker, bars in all_data.items():
        start_idx = step * BAR_COUNT
        end_idx = start_idx + BAR_COUNT
        
        if start_idx < len(bars):
            result[ticker] = bars[start_idx:end_idx]
        else:
            result[ticker] = None  # No more data available
    
    return result

def read_stock_values_v2(step):
    """
    Read the next BAR_COUNT bars for each ticker starting from the given step.
    
    Parameters:
    step (int): The starting index (0-based) for reading bars.
    
    Returns:
    dict: Dictionary with ticker symbols as keys and list of 4 bars as values.
          Returns None for tickers that don't have enough data.
    """
    with open(DATA_DIR / 'stock_values.json', 'r') as f:
        all_data = json.load(f)
    
    result = {}
    for ticker, bars in all_data.items():
        start_idx = step
        end_idx = start_idx + BAR_COUNT
        
        if start_idx < len(bars):
            result[ticker] = bars[start_idx:end_idx]
        else:
            result[ticker] = None  # No more data available
    
    return result

def read_stock_news(timestamp):
    """
    Read the last NEWS_COUNT news articles before the given timestamp for each ticker.
    
    Parameters:
    timestamp (str or int): The timestamp (Unix timestamp or datetime string) to filter news articles.
    
    Returns:
    dict: Dictionary with ticker symbols as keys and list of up to NEWS_COUNT news articles as values.
          Articles are sorted by datetime in descending order (most recent first).
    """
    # Convert timestamp string to Unix timestamp if needed
    if isinstance(timestamp, str):
        from datetime import datetime
        dt = datetime.fromisoformat(timestamp.replace('-05:00', '').replace('-04:00', ''))
        timestamp = int(dt.timestamp())
    
    with open(DATA_DIR / 'stock_news.json', 'r') as f:
        all_news = json.load(f)
    
    result = {}
    for ticker, articles in all_news.items():
        # Filter articles before the given timestamp
        filtered_articles = [
            article for article in articles 
            if article['datetime'] <= timestamp
        ]
        # Sort by datetime descending and take last NEWS_COUNT
        filtered_articles.sort(key=lambda x: x['datetime'], reverse=True)
        result[ticker] = filtered_articles[:NEWS_COUNT]
    
    return result

import json
from datetime import datetime, timezone

def read_stock_news_v2(
    cutoff_date_str: str,
    max_items: int = 100,
):
    """
    cutoff_date_str format: YYYY-MM-DD
    """
    # Convert cutoff date to UNIX timestamp (start of that day, UTC)
    cutoff_dt = datetime.strptime(cutoff_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    cutoff_ts = int(cutoff_dt.timestamp())
    with open(DATA_DIR / 'stock_news.json', "r") as f:
        data = json.load(f)

    collected = []

    for ticker, news_list in data.items():
        for item in news_list:
            if item.get("datetime", 0) < cutoff_ts:
                collected.append(item)

    # Sort by datetime descending (most recent first)
    collected.sort(key=lambda x: x["datetime"], reverse=True)

    # Return only the requested number
    return collected[:max_items]


if __name__ == "__main__":
    # Example: Read first 4 bars (step 0)
    step_0_data = read_stock_values(0)
    print("Step 0 (bars 0-3):")
    for ticker, bars in step_0_data.items():
        if bars:
            print(f"\n{ticker}: {len(bars)} bars")
            print(f"  First bar: {bars[0]['Datetime']}")
            print(f"  Last bar: {bars[-1]['Datetime']}")
    
    # Example: Read next 4 bars (step 1)
    print("\n" + "="*50)
    step_1_data = read_stock_values(1)
    print("\nStep 1 (bars 4-7):")
    for ticker, bars in step_1_data.items():
        if bars:
            print(f"\n{ticker}: {len(bars)} bars")
            print(f"  First bar: {bars[0]['Datetime']}")
            print(f"  Last bar: {bars[-1]['Datetime']}")
    
    # Example: Read news before a specific timestamp
    print("\n" + "="*50)
    print("\nNews before 2025-12-09 12:00:00:")
    news_data = read_stock_news("2025-12-09 12:00:00")
    for ticker, articles in news_data.items():
        print(f"\n{ticker}: {len(articles)} articles")
        for i, article in enumerate(articles, 1):
            from datetime import datetime
            dt = datetime.fromtimestamp(article['datetime'])
            print(f"  {i}. [{dt}] {article['headline'][:60]}...")
