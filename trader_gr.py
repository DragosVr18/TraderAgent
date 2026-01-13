from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from typing import Annotated, List, TypedDict, Dict, Any
from langgraph.graph import StateGraph, END, START, add_messages
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
import requests
import json
import re
from pathlib import Path

from config.config_reader import TOOLS_API_URL, SYSTEM_PROMPT, get_parameterized_system_prompt


class TradingAgent:
    """
    Auto-trading agent that predicts stock values and executes trades.
    """
    
    def __init__(self, 
                 base_url="http://localhost:11434/v1",
                 api_key="ollama",
                 llm_model="deepseek-r1:7b",
                 tools_api_url=TOOLS_API_URL):
        """
        Initialize the trading agent.
        
        Args:
            base_url: LLM API base URL
            api_key: API key for LLM
            llm_model: Model name to use
            tools_api_url: API URL for stock prediction tools
        """
        self.llm = ChatOpenAI(
            base_url=base_url,
            api_key=api_key,
            temperature=0,
            model=llm_model,
        )
        
        self.tools_api_url = tools_api_url
        self.stock_values_endpoint = f"{tools_api_url}/valuepredict"

        self.news_sentiment_endpoint = f"{tools_api_url}/analyzenews"
        
        # State variables
        self.stock_predictions = {}
        self.stock_current = {}
        self.fetch_values = True

        self.fetch_news = True
        self.portfolio_file = None
        self.trade_history = []
        
        # Build the agent graph
        self._build_graph()


    def _news_sentiment_tool(self, ticker: str) -> str:
        """
        Analyze news sentiment for a given ticker.
        
        Args:
            ticker: Stock ticker symbol
        Returns:
            str: Sentiment analysis result or error message
        """
        if self.fetch_news:
            try:
                response = requests.get(f"{self.news_sentiment_endpoint}")
                if response.status_code == 200:
                    data = response.json()
                    sentiment = data.get("sentiment", "No sentiment data available.")
                    self.sentiments = data.get("sentiments", {})
                    return f"News sentiment for {ticker}: {sentiment}"
                else:
                    return f"Error fetching news sentiment: {response.status_code}"
            except Exception as e:
                return f"Error connecting to news API: {str(e)}"

        if ticker in self.sentiments:
            sentiment = self.sentiments[ticker]


            #NOTE: Aici de facut modificarile in continuare...
    
    def _stock_value_prediction_tool(self, ticker: str) -> str:
        """
        Predict stock candle values for a given ticker.
        
        Args:
            ticker: Stock ticker symbol
        Returns:
            str: Predicted candle values or error message
        """
        if self.fetch_values:
            try:
                response = requests.get(self.stock_values_endpoint)
                if response.status_code == 200:
                    data = response.json()
                    self.stock_predictions = data.get("predictions", {})
                    self.stock_current = data.get("current", {})
                    self.fetch_values = False
                else:
                    return f"Error fetching stock values: {response.status_code}"
            except Exception as e:
                return f"Error connecting to stock API: {str(e)}"
        
        if ticker in self.stock_predictions:
            prediction = self.stock_predictions[ticker]
            current = self.stock_current.get(ticker, {})
            return (f"Predicted values for {ticker} - Open: {prediction['Open']}, High: {prediction['High']}, "
                    f"Low: {prediction['Low']}, Close: {prediction['Close']}, Volume: {prediction['Volume']}. "
                    f"Current values - Open: {current.get('Open', 'N/A')}, High: {current.get('High', 'N/A')}, "
                    f"Low: {current.get('Low', 'N/A')}, Close: {current.get('Close', 'N/A')}, Volume: {current.get('Volume', 'N/A')}.")
        else:
            return f"No prediction available for ticker: {ticker}"
    
    def _update_portfolio_tool(self, ticker: str, action: str, quantity: int) -> str:
        """
        Update portfolio by buying or selling stocks.
        
        Args:
            ticker: Stock ticker symbol
            action: 'buy' or 'sell'
            quantity: Number of shares to buy or sell
        Returns:
            str: Confirmation message
        """
        if action not in ['buy', 'sell']:
            return "Invalid action. Please specify 'buy' or 'sell'."
        
        try:
            with open(self.portfolio_file, 'r') as file:
                portfolio = json.load(file)
        except FileNotFoundError:
            portfolio = {"budget": 0, "stocks": {}}
        
        current_quantity = portfolio.get('stocks', {}).get(ticker, 0)
        budget = portfolio.get('budget', 0)
        price_per_share = self.stock_current.get(ticker, {}).get('Close', 0)
        total_price = price_per_share * quantity
        
        if action == 'buy':
            if total_price > budget:
                return f"Insufficient budget to buy {quantity} shares of {ticker}."
            portfolio.setdefault('stocks', {})[ticker] = current_quantity + quantity
            portfolio['budget'] = budget - total_price
            message = f"Bought {quantity} shares of {ticker} at ${price_per_share:.2f} each (Total: ${total_price:.2f})"
        else:  # sell
            if quantity > current_quantity:
                return f"Insufficient shares to sell {quantity} shares of {ticker}."
            portfolio['stocks'][ticker] = current_quantity - quantity
            portfolio['budget'] = budget + total_price
            message = f"Sold {quantity} shares of {ticker} at ${price_per_share:.2f} each (Total: ${total_price:.2f})"
        
        # Save portfolio
        with open(self.portfolio_file, 'w') as file:
            json.dump(portfolio, file, indent=4)
        
        # Add to trade history
        self.trade_history.append({
            "ticker": ticker,
            "action": action,
            "quantity": quantity,
            "price": price_per_share,
            "total": total_price,
            "message": message
        })
        
        return message
    
    def _parse_tool_call(self, text: str):
        """Parse tool calls from LLM text response using ReAct-style format."""
        tool_pattern = r'(stock_value_prediction|update_portolio)\s*\(\s*["\']([^"\']+)["\']\s*(?:,\s*["\']([^"\']+)["\']\s*)?(?:,\s*(\d+)\s*)?\)'
        matches = re.findall(tool_pattern, text)
        return matches
    
    def _predict_node(self, state: Dict) -> Dict:
        """Phase 1: Get predictions for all stocks."""
        last_message = state["messages"][-1] if state["messages"] else None
        if last_message and "Tool Result" in str(last_message.content):
            return {"phase": "decide"}
        
        response = self.llm.invoke(state["messages"])
        
        tool_calls = self._parse_tool_call(response.content)
        prediction_calls = [(name, arg1, arg2, arg3) for name, arg1, arg2, arg3 in tool_calls 
                            if name == "stock_value_prediction"]
        
        if prediction_calls:
            results = []
            for tool_name, ticker, _, _ in prediction_calls:
                result = self._stock_value_prediction_tool(ticker)
                results.append(f"[{ticker}] {result}")
            
            combined_results = "\n".join(results)
            return {
                "messages": [response, AIMessage(content=f"Tool Results:\n{combined_results}")],
                "phase": "decide"
            }
        
        return {"messages": [response], "phase": "decide"}
    
    def _decide_node(self, state: Dict) -> Dict:
        """Phase 2: Agent analyzes predictions and decides on trades."""
        decision_prompt = HumanMessage(content="""NOW execute trades based on the predictions.

You MUST call update_portolio tool for stocks you want to trade. Do NOT explain, just write the tool calls:

update_portolio("TICKER", "buy", quantity)
update_portolio("TICKER", "sell", quantity)

Write the actual tool calls now:""")
        
        response = self.llm.invoke(state["messages"] + [decision_prompt])
        
        tool_calls = self._parse_tool_call(response.content)
        update_calls = [(name, arg1, arg2, arg3) for name, arg1, arg2, arg3 in tool_calls 
                        if name == "update_portolio" and arg2 and arg3]
        
        if update_calls:
            results = []
            for tool_name, ticker, action, quantity in update_calls:
                result = self._update_portfolio_tool(ticker, action, int(quantity))
                results.append(f"[{ticker}] {result}")
            
            combined_results = "\n".join(results)
            return {
                "messages": [decision_prompt, response, AIMessage(content=f"Trade Results:\n{combined_results}")],
                "phase": "done"
            }
        
        return {"messages": [decision_prompt, response], "phase": "done"}
    
    def _route_phase(self, state: Dict) -> str:
        """Route to appropriate phase based on state."""
        phase = state.get("phase", "predict")
        
        if phase == "predict":
            return "predict"
        elif phase == "decide":
            return "decide"
        else:
            return "end"
    
    def _build_graph(self):
        """Build the LangGraph agent graph."""
        class AgentState(TypedDict):
            messages: Annotated[List[BaseMessage], add_messages]
            phase: str
        
        graph_builder = StateGraph(AgentState)
        
        graph_builder.add_node("predict", self._predict_node)
        graph_builder.add_node("decide", self._decide_node)
        
        graph_builder.add_edge(START, "predict")
        
        graph_builder.add_conditional_edges(
            "predict",
            self._route_phase,
            {
                "predict": "predict",
                "decide": "decide",
                "end": END,
            },
        )
        
        graph_builder.add_conditional_edges(
            "decide",
            self._route_phase,
            {
                "predict": "predict",
                "decide": "decide",
                "end": END,
            },
        )
        
        self.agent_graph = graph_builder.compile()
    
    def setup_portfolio(self, budget: float, stocks: Dict[str, float], portfolio_file: str = "temp_portfolio.json"):
        """
        Setup the portfolio for trading.
        
        Args:
            budget: Available budget
            stocks: Dictionary of ticker -> quantity
            portfolio_file: Path to save portfolio
        """
        self.portfolio_file = portfolio_file
        portfolio = {
            "budget": budget,
            "stocks": stocks
        }
        
        with open(portfolio_file, 'w') as file:
            json.dump(portfolio, file, indent=4)
    
    def load_portfolio_from_file(self, filepath: str):
        """
        Load portfolio from JSON file.
        
        Args:
            filepath: Path to portfolio JSON file
        """
        self.portfolio_file = filepath
        with open(filepath, 'r') as file:
            portfolio = json.load(file)
        return portfolio
    
    def run_iteration(self, strategy: str) -> Dict[str, Any]:
        """
        Run a single trading iteration.
        
        Args:
            strategy: Trading strategy (e.g., "high risk", "low risk - long term")
        
        Returns:
            dict: Results including trade history and final portfolio
        """
        # Reset fetch_values flag at the start of each iteration
        self.fetch_values = True
        self.fetch_news = True
        self.trade_history = []
        
        # Load current portfolio
        with open(self.portfolio_file, 'r') as file:
            portfolio = json.load(file)
        
        stocks_list = list(portfolio.get('stocks', {}).keys())
        
        # Create tool call examples
        tool_examples = "\n".join([f'stock_value_prediction("{ticker}")' for ticker in stocks_list])
        
        # Add strategy to system prompt
        strategy_prompt = f"\n\nTrading Strategy: {strategy}\nYou must follow this strategy when making trading decisions."
        
        user_prompt = f"""Current Portfolio: {portfolio}

You must analyze ALL these stocks: {stocks_list}

Call the stock_value_prediction tool for each ticker like this:
{tool_examples}

Do this now."""
        
        initial_state = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT + strategy_prompt),
                HumanMessage(content=user_prompt),
            ],
            "phase": "predict"
        }
        
        final_state = self.agent_graph.invoke(initial_state, config={"recursion_limit": 10})
        
        # Load final portfolio state
        with open(self.portfolio_file, 'r') as file:
            final_portfolio = json.load(file)
        
        return {
            "trade_history": self.trade_history,
            "initial_portfolio": portfolio,
            "final_portfolio": final_portfolio,
            "messages": final_state["messages"]
        }
    
    def run_multiple_iterations(self, num_iterations: int, strategy: str, progress_callback=None):
        """
        Run multiple trading iterations.
        
        Args:
            num_iterations: Number of iterations to run
            strategy: Trading strategy
            progress_callback: Optional callback function for progress updates
        
        Returns:
            list: Results from all iterations
        """
        all_results = []
        
        for i in range(1, num_iterations + 1):
            # Reset fetch_values flag before each iteration
            self.fetch_values = True
            self.fetch_news = True
            
            if progress_callback:
                progress_callback(i, num_iterations, f"Running iteration {i}/{num_iterations}")
            
            result = self.run_iteration(strategy)
            result["iteration"] = i
            all_results.append(result)
        
        return all_results