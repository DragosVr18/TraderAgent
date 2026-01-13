import json
from pathlib import Path
from datetime import date, timedelta


import sys
sys.path.append("/teamspace/studios/this_studio/TraderAgent")

from data.fetch_news import fetch_stock_news
from data.fetch_value import fetch_stock_value
from config import API_KEY

NO_DAYS = 80
INTERVAL = '1d'

def aggregate_stock_data():
    # Load stocks from user portfolio
    portfolio_path = Path(__file__).parent.parent / 'user_prtf.json'
    with open(portfolio_path, 'r') as f:
        portfolio = json.load(f)

    STOCKS = list(portfolio['stocks'].keys())

    stock_news = {}
    stock_values = {}

    to_date = date.today()
    from_date = to_date - timedelta(days=NO_DAYS)

    for symbol in STOCKS:
        # Fetch news articles
        news_articles = fetch_stock_news(
            API_KEY,
            symbol,
            count=None,
            from_date=str(from_date),
            to_date=str(to_date)
        )
        print(len(news_articles))
        stock_news[symbol] = news_articles

        # Fetch stock values
        stock_data = fetch_stock_value(
           symbol,
           count=None,
           interval=INTERVAL,
           period=f"{NO_DAYS}d"
        )
        stock_values[symbol] = stock_data

    # Save aggregated data to JSON files
    output_path = Path(__file__).parent.parent / 'data_aggregated_v4'
    output_path.mkdir(exist_ok=True)
    # with open(output_path / 'stock_news.json', 'w') as f:
    #     json.dump(stock_news, f, default=str, indent=4)
    with open(output_path / 'stock_values.json', 'w') as f:
       json.dump({k: v.reset_index().to_dict(orient='records') for k, v in stock_values.items()}, f, default=str, indent=4)


if __name__ == "__main__":
    aggregate_stock_data()
