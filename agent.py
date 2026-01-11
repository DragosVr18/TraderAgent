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
NUM_ITERATIONS = 1
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
    phase: str  # Track current phase: 'predict', 'decide', or 'done'

def parse_tool_call(text: str):
    """Parse tool calls from LLM text response using ReAct-style format."""
    # Look for patterns like: stock_value_prediction("AAPL") or update_portolio("AAPL", "buy", 10)
    tool_pattern = r'(stock_value_prediction|update_portolio)\s*\(\s*["\']([^"\']+)["\']\s*(?:,\s*["\']([^"\']+)["\']\s*)?(?:,\s*(\d+)\s*)?\)'
    matches = re.findall(tool_pattern, text)
    return matches

def predict_node(state: AgentState) -> AgentState:
    """Phase 1: Get predictions for all stocks."""
    # Check if we already have predictions
    last_message = state["messages"][-1] if state["messages"] else None
    if last_message and "Tool Result" in str(last_message.content):
        # Already have predictions, move to decide phase
        return {"phase": "decide"}
    
    response = llm.invoke(state["messages"])
    
    # Parse and execute ONLY stock_value_prediction tools
    tool_calls = parse_tool_call(response.content)
    prediction_calls = [(name, arg1, arg2, arg3) for name, arg1, arg2, arg3 in tool_calls 
                        if name == "stock_value_prediction"]
    
    if prediction_calls:
        print(f"\n[PREDICT PHASE] Found {len(prediction_calls)} prediction call(s)")
        results = []
        for tool_name, ticker, _, _ in prediction_calls:
            print(f"[PREDICT PHASE] Executing: {tool_name}({ticker})")
            result = stock_value_prediction.invoke({"ticker": ticker})
            results.append(f"[{ticker}] {result}")
        
        combined_results = "\n".join(results)
        return {
            "messages": [response, AIMessage(content=f"Tool Results:\n{combined_results}")],
            "phase": "decide"
        }
    
    return {"messages": [response], "phase": "decide"}

def decide_node(state: AgentState) -> AgentState:
    """Phase 2: Agent analyzes predictions and decides on trades."""
    # Add a very directive message forcing tool usage
    decision_prompt = HumanMessage(content="""NOW execute trades based on the predictions.

You MUST call update_portolio tool for stocks you want to trade. Do NOT explain, just write the tool calls:

update_portolio("TICKER", "buy", quantity)
update_portolio("TICKER", "sell", quantity)

Write the actual tool calls now:""")
    
    response = llm.invoke(state["messages"] + [decision_prompt])
    
    # Parse and execute ONLY update_portolio tools
    tool_calls = parse_tool_call(response.content)
    update_calls = [(name, arg1, arg2, arg3) for name, arg1, arg2, arg3 in tool_calls 
                    if name == "update_portolio" and arg2 and arg3]
    
    if update_calls:
        print(f"\n[EXECUTE PHASE] Found {len(update_calls)} trade call(s)")
        results = []
        for tool_name, ticker, action, quantity in update_calls:
            print(f"[EXECUTE PHASE] Executing: {tool_name}({ticker}, {action}, {quantity})")
            result = update_portolio.invoke({"ticker": ticker, "action": action, "quantity": int(quantity)})
            results.append(f"[{ticker}] {result}")
        
        combined_results = "\n".join(results)
        return {
            "messages": [decision_prompt, response, AIMessage(content=f"Trade Results:\n{combined_results}")],
            "phase": "done"
        }
    
    print("\n[EXECUTE PHASE] WARNING: No trade calls found in response!")
    print(f"[EXECUTE PHASE] Response was: {response.content[:200]}...")
    return {"messages": [decision_prompt, response], "phase": "done"}

def route_phase(state: AgentState) -> str:
    """Route to appropriate phase based on state."""
    phase = state.get("phase", "predict")
    
    if phase == "predict":
        return "predict"
    elif phase == "decide":
        return "decide"
    else:
        return "end"

graph_builder = StateGraph(AgentState)

# Add nodes for each phase
graph_builder.add_node("predict", predict_node)
graph_builder.add_node("decide", decide_node)

# Start at predict phase
graph_builder.add_edge(START, "predict")

# Route based on phase
graph_builder.add_conditional_edges(
    "predict",
    route_phase,
    {
        "predict": "predict",
        "decide": "decide",
        "end": END,
    },
)

graph_builder.add_conditional_edges(
    "decide",
    route_phase,
    {
        "predict": "predict",
        "decide": "decide",
        "end": END,
    },
)

agent_graph = graph_builder.compile()

def run_agent(user_input: str):
    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_input),
        ],
        "phase": "predict"
    }
    final_state = agent_graph.invoke(initial_state, config={"recursion_limit": 10})
    return final_state

def run_trading_iteration(iteration_num: int):
    """Run a single trading iteration."""
    global FETCH_VALUES
    
    print(f"\n{'='*60}")
    print(f"ITERATION {iteration_num}")
    print(f"{'='*60}\n")
    
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

    # Reset the fetch flag for next iteration
    FETCH_VALUES = True
    
    return result

# Main execution loop
if __name__ == "__main__":
    print(f"\nStarting trading agent for {NUM_ITERATIONS} iterations...\n")
    
    for i in range(1, NUM_ITERATIONS + 1):
        run_trading_iteration(i)
    
    print(f"\n{'='*60}")
    print(f"Completed all {NUM_ITERATIONS} iterations")
    print(f"{'='*60}\n")

