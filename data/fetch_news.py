import finnhub
from datetime import date
from datetime import datetime, timedelta
import time

def fetch_stock_news(api_key, symbol, count=None, from_date=date.today(), to_date=date.today(), chunk_days=15):
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
    # finnhub_client = finnhub.Client(api_key=api_key)
    # news = finnhub_client.company_news(symbol, _from=from_date, to=to_date)
    # if count is not None and count <= len(news):
    #     return news[:count]
    # else:
    #     return news
    finnhub_client = finnhub.Client(api_key=api_key)
    all_news = []

    # Convert strings to datetime
    current_start = datetime.strptime(from_date, "%Y-%m-%d")
    end_date_dt = datetime.strptime(to_date, "%Y-%m-%d")

    while current_start <= end_date_dt:
        current_end = min(end_date_dt, current_start + timedelta(days=chunk_days))
        chunk_news = finnhub_client.company_news(
            symbol,
            _from=current_start.strftime("%Y-%m-%d"),
            to=current_end.strftime("%Y-%m-%d")
        )
        all_news.extend(chunk_news)
        current_start = current_end + timedelta(days=1)

        # Optional early break if count is reached
        if count is not None and len(all_news) >= count:
            return all_news[:count]
        time.sleep(1)  #
        print("Tot al news articles fetched so far:", len(all_news))

    # Sort by datetime descending (most recent first)
    all_news.sort(key=lambda x: x["datetime"], reverse=True)

    # Apply count limit if needed
    if count is not None:
        return all_news[:count]
    return all_news
    
if __name__ == "__main__":
    import sys
    sys.path.append("/teamspace/studios/this_studio/TraderAgent")
    from config import API_KEY

    symbol = "AAPL"
    news_articles = fetch_stock_news(API_KEY, symbol, count=5, from_date="2025-06-01", to_date="2025-06-10")
    for article in news_articles:
        print(f"Headline: {article['headline']}")
        print(f"Date: {article['datetime']}")
        print(f"Summary: {article['summary']}")
        print(f"URL: {article['url']}\n")