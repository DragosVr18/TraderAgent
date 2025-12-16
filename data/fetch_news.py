import finnhub
from datetime import date

def fetch_stock_news(api_key, symbol, count=None, from_date=date.today(), to_date=date.today()):
    """
    Fetch the latest news articles for a given stock symbol.

    Parameters:
    api_key (str): Finnhub API key.
    symbol (str): Stock symbol to fetch news for.
    count (int): Number of news articles to fetch.
    from_date (str): Start date for news in YYYY-MM-DD format.
    to_date (str): End date for news in YYYY-MM-DD format.

    Returns:
    list: A list of news articles.
    """
    finnhub_client = finnhub.Client(api_key=api_key)
    news = finnhub_client.company_news(symbol, _from=from_date, to=to_date)
    if count is not None and count <= len(news):
        return news[:count]
    else:
        return news
    
if __name__ == "__main__":
    from config import API_KEY

    symbol = "AAPL"
    news_articles = fetch_stock_news(API_KEY, symbol, count=5, from_date="2025-06-01", to_date="2025-06-10")
    for article in news_articles:
        print(f"Headline: {article['headline']}")
        print(f"Date: {article['datetime']}")
        print(f"Summary: {article['summary']}")
        print(f"URL: {article['url']}\n")