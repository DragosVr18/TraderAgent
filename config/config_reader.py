import yaml
from pathlib import Path


def load_config(config_path: str = None):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. Defaults to config/config.yaml
        
    Returns:
        dict: Configuration dictionary
    """
    if config_path is None:
        # Get the config file relative to this script's location
        config_path = Path(__file__).parent / "config.yaml"
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config


config = load_config()

API_KEY = config['api']['finnhub_key']
FETCH_INTERVAL = config['fetch_interval']
TICKER_INTERVAL = config.get('ticker_interval', '5-minute')
TOOLS_API_URL = config['tools_url']
SYSTEM_PROMPT = config['system_prompt']

def get_parameterized_system_prompt(strategy: str) -> str:
    """
    Get system prompt with parameters filled in.
    
    Args:
        ticker_interval: Time interval for tickers (e.g., "5-minute", "1-hour")
        strategy: Trading strategy to apply
    
    Returns:
        str: Parameterized system prompt
    """
    prompt = SYSTEM_PROMPT
    
    # Replace placeholders
    prompt = prompt.replace('{ticker_interval}', TICKER_INTERVAL)
    prompt = prompt.replace('{strategy}', strategy)
    
    return prompt


if __name__ == "__main__":
    print(f"API Key: {API_KEY}")
    print(f"Fetch Interval: {FETCH_INTERVAL}")