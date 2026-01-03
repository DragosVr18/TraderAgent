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
TOOLS_API_URL = config['tools_url']
SYSTEM_PROMPT = config['system_prompt']


if __name__ == "__main__":
    print(f"API Key: {API_KEY}")
    print(f"Fetch Interval: {FETCH_INTERVAL}")