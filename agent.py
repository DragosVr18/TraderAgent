from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from typing import Annotated, List, TypedDict
from langgraph.graph import StateGraph, END, START, add_messages
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
import requests
import json
import re

from config.config_reader import TOOLS_API_URL, SYSTEM_PROMPT

base_url = "http://localhost:1234/v1"
api_key = "lm-studio"
llm_model = "deepseek-r1-distill-qwen-7b"

llm = ChatOpenAI(
    base_url=base_url,
    api_key=api_key,
    temperature=0,
    model=llm_model,
)

FETCH_VALUES = True
STOCK_VALUES_ENDPOINT = f"{TOOLS_API_URL}/valuepredict"
stock_predictions = {}
stock_current = {}

@tool
def stock_value_prediction(ticker: str) -> str:
    """
    Predict stock candle values for a given ticker.
    Args:
        ticker: Stock ticker symbol
    Returns:
        str: Predicted candle values or error message
    """
    global FETCH_VALUES
    global stock_predictions
    global stock_current
    if FETCH_VALUES:
        response = requests.get(STOCK_VALUES_ENDPOINT)
        if response.status_code == 200:
            data = response.json()
            stock_predictions = data.get("predictions", {})
            stock_current = data.get("current", {})
            FETCH_VALUES = False
        else:
            return f"Error fetching stock values: {response.status_code}"
    if ticker in stock_predictions:
        prediction = stock_predictions[ticker]
        current = stock_current.get(ticker, {})
        return (f"Predicted values for {ticker} - Open: {prediction['Open']}, High: {prediction['High']}, "
                f"Low: {prediction['Low']}, Close: {prediction['Close']}, Volume: {prediction['Volume']}. "
                f"Current values - Open: {current.get('Open', 'N/A')}, High: {current.get('High', 'N/A')}, "
                f"Low: {current.get('Low', 'N/A')}, Close: {current.get('Close', 'N/A')}, Volume: {current.get('Volume', 'N/A')}.")
    else:
        return f"No prediction available for ticker: {ticker}"
    
@tool
def update_portolio(ticker: str, action: str, quantity: int) -> str:
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
        with open('user_prtf.json', 'r') as file:
            portfolio = json.load(file)
    except FileNotFoundError:
        portfolio = {}
    current_quantity = portfolio.get('stocks', {}).get(ticker, 0)
    budget = portfolio.get('budget', 0)
    price_per_share = stock_current.get(ticker, {}).get('Close', 0)
    total_price = price_per_share * quantity
    if action == 'buy':
        if total_price > budget:
            return f"Insufficient budget to buy {quantity} shares of {ticker}."
        portfolio.setdefault('stocks', {})[ticker] = current_quantity + quantity
        portfolio['budget'] = budget - total_price
        message = f"Bought {quantity} shares of {ticker}."
    else:  # sell
        if quantity > current_quantity:
            return f"Insufficient shares to sell {quantity} shares of {ticker}."
        portfolio['stocks'][ticker] = current_quantity - quantity
        portfolio['budget'] = budget + total_price
        message = f"Sold {quantity} shares of {ticker}."

    with open('user_prtf.json', 'w') as file:
        json.dump(portfolio, file, indent=4)

    return message

tools = [stock_value_prediction, update_portolio]
tools_dict = {tool.name: tool for tool in tools}

class AgentState(TypedDict):
    """State for the LangGraph agent: a list of messages."""
    messages: Annotated[List[BaseMessage], add_messages]

def parse_tool_call(text: str):
    """Parse tool calls from LLM text response using ReAct-style format."""
    # Look for patterns like: stock_value_prediction("AAPL") or update_portolio("AAPL", "buy", 10)
    tool_pattern = r'(stock_value_prediction|update_portolio)\s*\(\s*["\']([^"\']+)["\']\s*(?:,\s*["\']([^"\']+)["\']\s*)?(?:,\s*(\d+)\s*)?\)'
    matches = re.findall(tool_pattern, text)
    return matches

def agent_node(state: AgentState) -> AgentState:
    """Call the LLM and parse for tool calls in the response text."""
    response = llm.invoke(state["messages"])
    
    # Check if the response contains tool call instructions
    tool_calls = parse_tool_call(response.content)
    
    if tool_calls:
        print(f"\n[DEBUG] Found {len(tool_calls)} tool call(s) in response")
        # Execute ALL tool calls found
        results = []
        for tool_name, arg1, arg2, arg3 in tool_calls:
            print(f"[DEBUG] Executing: {tool_name}({arg1}, {arg2}, {arg3})")
            
            if tool_name == "stock_value_prediction":
                result = stock_value_prediction.invoke({"ticker": arg1})
                results.append(f"[{arg1}] {result}")
            elif tool_name == "update_portolio" and arg2 and arg3:
                result = update_portolio.invoke({"ticker": arg1, "action": arg2, "quantity": int(arg3)})
                results.append(f"[{arg1}] {result}")
            else:
                results.append(f"Error: Invalid tool call for {tool_name}")
        
        # Add the tool results as a message
        combined_results = "\n".join(results)
        return {"messages": [response, AIMessage(content=f"Tool Results:\n{combined_results}")]}
    
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    """Determine if we should continue or end."""
    last_message = state["messages"][-1]
    
    # Check if we've done too many iterations
    if len(state["messages"]) > 30:
        return "end"
    
    # Check if last message contains tool calls - if so, continue
    if isinstance(last_message, AIMessage) and last_message.content:
        if parse_tool_call(last_message.content):
            return "continue"
    
    # If the last message contains tool results, continue to let agent respond
    if last_message.content and ("Tool Result" in last_message.content):
        return "continue"
    
    # If no tool calls and no tool results, we're probably done
    return "end"

graph_builder = StateGraph(AgentState)

graph_builder.add_node("agent", agent_node)

graph_builder.add_edge(START, "agent")

graph_builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    },
)

agent_graph = graph_builder.compile()

def run_agent(user_input: str):
    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_input),
        ]
    }
    final_state = agent_graph.invoke(initial_state, config={"recursion_limit": 10})
    return final_state

portofolio = json.load(open('user_prtf.json'))
stocks_list = list(portofolio.get('stocks', {}).keys())

# Create explicit tool call examples for all stocks
tool_examples = "\n".join([f'stock_value_prediction("{ticker}")' for ticker in stocks_list])

user_prompt = f"""Current Portfolio: {portofolio}

You must analyze ALL these stocks: {stocks_list}

Call the stock_value_prediction tool for each ticker like this:
{tool_examples}

Do this now."""

result = run_agent(user_prompt)
for m in result["messages"]:
    print(f"[{m.type}] {m.content}")

FETCH_VALUES = True

