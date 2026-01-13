import yaml
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
import requests
import json
import os
from typing import List, Dict, Optional

# -------------------------
# LLM Wrapper Class
# -------------------------
class StockLLMAgent:
    def __init__(self, model_name: str, base_url: str, api_key: str, prompt_yaml_path: str, temperature: float = 0.0):
        """
        Initialize the Stock LLM Agent.

        Args:
            model_name: Model identifier (e.g., "deepseek-r1:7b")
            base_url: Ollama base URL
            api_key: Ollama API key (required by LangChain)
            prompt_yaml_path: Path to YAML file containing system prompt
            temperature: Sampling temperature for LLM
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.llm = None
        self.system_prompt = None
        self._load_prompt(prompt_yaml_path)
        self._init_llm()
        
    def _load_prompt(self, yaml_path: str):
        """Load system prompt from YAML file."""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Prompt YAML file not found: {yaml_path}")
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        self.system_prompt = data.get("system_prompt", "")
        if not self.system_prompt:
            raise ValueError("No 'system_prompt' key found in YAML file.")
        
    def _init_llm(self):
        """Initialize the ChatOpenAI LLM client."""
        self.llm = ChatOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model_name,
            temperature=self.temperature,
        )
    
    def run(self, user_input: str) -> str:
        """
        Invoke the LLM with a human message and system prompt.
        
        Args:
            user_input: Text to send to the model
        
        Returns:
            Model response text
        """
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_input),
        ]
        response = self.llm.invoke(messages)
        return response.content

