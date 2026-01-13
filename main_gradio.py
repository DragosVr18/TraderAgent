import gradio as gr
import json
import pandas as pd
from pathlib import Path
import tempfile
from trader_gr import TradingAgent

# Initialize the trading agent
agent = TradingAgent()

def format_trade_history(results):
    """Format trade history for display."""
    all_trades = []
    
    for result in results:
        iteration = result.get("iteration", 1)
        for trade in result.get("trade_history", []):
            all_trades.append({
                "Iteration": iteration,
                "Ticker": trade["ticker"],
                "Action": trade["action"].upper(),
                "Quantity": trade["quantity"],
                "Price": f"${trade['price']:.2f}",
                "Total": f"${trade['total']:.2f}",
                "Details": trade["message"]
            })
    
    if not all_trades:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_trades)
    return df

def format_portfolio_comparison(initial, final, stock_prices):
    """Format initial vs final portfolio comparison with total value."""
    # Calculate total values
    initial_cash = initial.get('budget', 0)
    final_cash = final.get('budget', 0)
    
    initial_stocks_value = 0
    final_stocks_value = 0
    
    for ticker, qty in initial.get('stocks', {}).items():
        price = stock_prices.get(ticker, {}).get('Close', 0)
        initial_stocks_value += qty * price
    
    for ticker, qty in final.get('stocks', {}).items():
        price = stock_prices.get(ticker, {}).get('Close', 0)
        final_stocks_value += qty * price
    
    initial_total = initial_cash + initial_stocks_value
    final_total = final_cash + final_stocks_value
    total_change = final_total - initial_total
    
    comparison = "### Portfolio Comparison\n\n"
    comparison += "#### Cash & Assets\n\n"
    comparison += f"**Initial Cash:** ${initial_cash:,.2f}\n"
    comparison += f"**Initial Stocks Value:** ${initial_stocks_value:,.2f}\n"
    comparison += f"**Initial Total Value:** ${initial_total:,.2f}\n\n"
    
    comparison += f"**Final Cash:** ${final_cash:,.2f}\n"
    comparison += f"**Final Stocks Value:** ${final_stocks_value:,.2f}\n"
    comparison += f"**Final Total Value:** ${final_total:,.2f}\n\n"
    
    change_symbol = "📈" if total_change >= 0 else "📉"
    comparison += f"{change_symbol} **Total Change:** ${total_change:,.2f} ({((total_change/initial_total)*100):.2f}%)\n\n"
    
    comparison += "#### Stock Holdings\n\n"
    comparison += "| Ticker | Initial | Final | Change | Current Price | Value Change |\n"
    comparison += "|--------|---------|-------|--------|---------------|-------------|\n"
    
    all_tickers = set(list(initial.get('stocks', {}).keys()) + list(final.get('stocks', {}).keys()))
    
    for ticker in sorted(all_tickers):
        init_qty = initial.get('stocks', {}).get(ticker, 0)
        final_qty = final.get('stocks', {}).get(ticker, 0)
        change = final_qty - init_qty
        change_str = f"+{change}" if change > 0 else str(change)
        
        price = stock_prices.get(ticker, {}).get('Close', 0)
        value_change = change * price
        value_change_str = f"+${value_change:.2f}" if value_change >= 0 else f"-${abs(value_change):.2f}"
        
        comparison += f"| {ticker} | {init_qty} | {final_qty} | {change_str} | ${price:.2f} | {value_change_str} |\n"
    
    return comparison

def run_trading(budget, stocks_df, json_file, strategy, num_iterations):
    """
    Main function to run the trading agent with live updates.
    """
    try:
        # Progress tracking
        # progress(0, desc="Setting up portfolio...")
        yield pd.DataFrame(), "", "⏳ Setting up portfolio...", ""
        
        # Determine portfolio source
        if json_file is not None:
            # Load from uploaded JSON file
            portfolio = agent.load_portfolio_from_file(json_file)
            budget = portfolio.get("budget", 0)
            stocks = portfolio.get("stocks", {})
            # progress(0.1, desc="Portfolio loaded from JSON file")
        else:
            # Parse manual inputs from dataframe
            if not budget or budget <= 0:
                yield pd.DataFrame(), "", "❌ Error: Please provide a valid budget.", ""
                return
            
            # Convert dataframe to stocks dictionary
            stocks = {}
            if stocks_df is not None and len(stocks_df) > 0:
                for _, row in stocks_df.iterrows():
                    ticker = str(row.get('Ticker', '')).strip().upper()
                    try:
                        quantity = float(row.get('Quantity', 0))
                        if ticker and quantity > 0:
                            stocks[ticker] = quantity
                    except (ValueError, TypeError):
                        continue
            
            if not stocks:
                yield pd.DataFrame(), "", "❌ Error: Please provide at least one stock holding.", ""
                return
            
            # Setup portfolio
            # progress(0.1, desc="Portfolio setup complete")
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            temp_file.close()
            agent.setup_portfolio(budget, stocks, temp_file.name)
        
        yield pd.DataFrame(), "", f"✅ Portfolio setup complete! Starting {num_iterations} iteration(s)...", ""
        
        # Run iterations with live updates
        all_results = []
        initial_portfolio = None
        initial_summary_shown = False
        
        for i in range(1, num_iterations + 1):
            # progress((0.1 + (i / num_iterations) * 0.8), desc=f"Running iteration {i}/{num_iterations}")
            
            # Run single iteration
            agent.fetch_values = True
            agent.trade_history = []
            
            with open(agent.portfolio_file, 'r') as file:
                portfolio = json.load(file)
            
            if initial_portfolio is None:
                initial_portfolio = portfolio.copy()
            
            stocks_list = list(portfolio.get('stocks', {}).keys())
            tool_examples = "\n".join([f'stock_value_prediction("{ticker}")' for ticker in stocks_list])
            strategy_prompt = f"\n\nTrading Strategy: {strategy}\nYou must follow this strategy when making trading decisions."
            
            user_prompt = f"""Current Portfolio: {portfolio}

You must analyze ALL these stocks: {stocks_list}

Call the stock_value_prediction tool for each ticker like this:
{tool_examples}

Do this now."""
            
            from langchain_core.messages import SystemMessage, HumanMessage
            from config.config_reader import SYSTEM_PROMPT
            
            initial_state = {
                "messages": [
                    SystemMessage(content=SYSTEM_PROMPT + strategy_prompt),
                    HumanMessage(content=user_prompt),
                ],
                "phase": "predict"
            }
            
            final_state = agent.agent_graph.invoke(initial_state, config={"recursion_limit": 10})
            
            with open(agent.portfolio_file, 'r') as file:
                final_portfolio = json.load(file)
            
            result = {
                "trade_history": agent.trade_history,
                "initial_portfolio": portfolio,
                "final_portfolio": final_portfolio,
                "messages": final_state["messages"],
                "iteration": i
            }
            all_results.append(result)
            
            # Show initial portfolio value before first iteration
            if not initial_summary_shown:
                initial_cash = initial_portfolio.get('budget', 0)
                initial_stocks_value = sum(
                    initial_portfolio.get('stocks', {}).get(ticker, 0) * agent.stock_current.get(ticker, {}).get('Close', 0)
                    for ticker in initial_portfolio.get('stocks', {}).keys()
                )
                initial_total = initial_cash + initial_stocks_value
                
                initial_display = f"""### Initial Portfolio Value

**Cash:** ${initial_cash:,.2f}
**Stocks Value:** ${initial_stocks_value:,.2f}
**Total Portfolio Value:** ${initial_total:,.2f}

---

"""
                initial_summary_shown = True
            else:
                initial_display = ""
            
            # Update display after each iteration
            trade_history_df = format_trade_history(all_results)
            total_trades = sum(len(r.get("trade_history", [])) for r in all_results)
            
            summary = initial_display + f"""### Trading Summary (After Iteration {i}/{num_iterations})
            
**Strategy:** {strategy}
**Iterations Completed:** {i}/{num_iterations}
**Total Trades Executed:** {total_trades}

{format_portfolio_comparison(initial_portfolio, final_portfolio, agent.stock_current)}
"""
            
            progress_text = f"⏳ Iteration {i}/{num_iterations} complete. {total_trades} trade(s) so far..."
            status = ""
            
            yield trade_history_df, summary, progress_text, status
        
        # Final update
        # progress(1.0, desc="Complete!")
        trade_history_df = format_trade_history(all_results)
        total_trades = sum(len(r.get("trade_history", [])) for r in all_results)
        
        # Calculate initial total value
        initial_cash = initial_portfolio.get('budget', 0)
        initial_stocks_value = sum(
            initial_portfolio.get('stocks', {}).get(ticker, 0) * agent.stock_current.get(ticker, {}).get('Close', 0)
            for ticker in initial_portfolio.get('stocks', {}).keys()
        )
        initial_total = initial_cash + initial_stocks_value
        
        summary = f"""### Initial Portfolio Value

**Cash:** ${initial_cash:,.2f}
**Stocks Value:** ${initial_stocks_value:,.2f}
**Total Portfolio Value:** ${initial_total:,.2f}

---

### Final Trading Summary
        
**Strategy:** {strategy}
**Iterations Completed:** {num_iterations}
**Total Trades Executed:** {total_trades}

{format_portfolio_comparison(initial_portfolio, final_portfolio, agent.stock_current)}
"""
        
        yield trade_history_df, summary, "✅ Trading completed successfully!", ""
        
    except Exception as e:
        yield pd.DataFrame(), "", f"❌ Trading failed: {str(e)}", ""

def load_json_preview(json_file):
    """Preview uploaded JSON file."""
    if json_file is None:
        return "No file uploaded"
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        preview = f"""**Budget:** ${data.get('budget', 0):,.2f}

**Stocks:**
"""
        for ticker, qty in data.get('stocks', {}).items():
            preview += f"- {ticker}: {qty} shares\n"
        
        return preview
    except Exception as e:
        return f"Error reading file: {str(e)}"

def add_stock_row(stocks_df):
    """Add a new row to the stocks dataframe."""
    if stocks_df is None or len(stocks_df) == 0:
        return pd.DataFrame({"Ticker": [""], "Quantity": [0.0]})
    
    new_row = pd.DataFrame({"Ticker": [""], "Quantity": [0.0]})
    return pd.concat([stocks_df, new_row], ignore_index=True)

def create_default_stocks():
    """Create default stocks dataframe."""
    return pd.DataFrame({
        "Ticker": ["AAPL", "MSFT", "GOOGL", "NFLX", "ORCL"],
        "Quantity": [0.56, 1.45, 2.0, 0.52, 1.34]
    })

# Create Gradio interface
with gr.Blocks(title="Auto-Trading Agent", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🤖 Auto-Trading Agent
    
    Configure your portfolio and let the AI agent make trading decisions based on stock predictions.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Portfolio Setup")
            
            with gr.Tab("Manual Input"):
                budget_input = gr.Number(
                    label="Budget ($)",
                    value=4000,
                    minimum=0,
                    info="Your available trading budget"
                )
                
                gr.Markdown("**Stock Holdings**")
                stocks_input = gr.Dataframe(
                    headers=["Ticker", "Quantity"],
                    datatype=["str", "number"],
                    row_count=3,
                    col_count=(2, "fixed"),
                    value=create_default_stocks(),
                    label="Enter your stocks",
                    interactive=True
                )
                
                add_row_btn = gr.Button("➕ Add Stock Row", size="sm")
                add_row_btn.click(
                    fn=add_stock_row,
                    inputs=[stocks_input],
                    outputs=[stocks_input]
                )
            
            with gr.Tab("Upload JSON"):
                json_file_input = gr.File(
                    label="Upload Portfolio JSON",
                    file_types=[".json"],
                    type="filepath"
                )
                
                json_preview = gr.Markdown("No file uploaded")
                json_file_input.change(
                    fn=load_json_preview,
                    inputs=[json_file_input],
                    outputs=[json_preview]
                )
            
            gr.Markdown("### Trading Configuration")
            
            strategy_input = gr.Radio(
                label="Trading Strategy",
                choices=[
                    "High Risk - Aggressive Growth",
                    "Medium Risk - Balanced",
                    "Low Risk - Long Term Conservative",
                    "Day Trading - Quick Profits",
                    "Value Investing - Undervalued Stocks"
                ],
                value="Medium Risk - Balanced",
                info="Select your preferred trading strategy"
            )
            
            iterations_input = gr.Slider(
                label="Number of Iterations",
                minimum=1,
                maximum=10,
                value=4,
                step=1,
                info="How many trading cycles to run"
            )
            
            run_button = gr.Button("🚀 Start Trading", variant="primary", size="lg")
            
            # NEW: Dedicated progress display
            # progress_output = gr.Textbox(
            #     label="Progress",
            #     lines=2,
            #     max_lines=3,
            #     interactive=False,
            #     show_label=True
            # )
            
            status_output = gr.Markdown("")
        
        with gr.Column(scale=2):
            gr.Markdown("### Trading Results")
            
            summary_output = gr.Markdown("")
            
            trade_history_output = gr.Dataframe(
                label="Trade History (Live Updates)",
                headers=["Iteration", "Ticker", "Action", "Quantity", "Price", "Total", "Details"],
                wrap=True,
                # visible=False  # Start hidden
            )
    
    # Connect the run button with streaming output
    run_button.click(
        fn=run_trading,
        inputs=[
            budget_input,
            stocks_input,
            json_file_input,
            strategy_input,
            iterations_input
        ],
        outputs=[
            trade_history_output,
            summary_output,
            # progress_output,  # NEW: Added progress output
            status_output
        ],
        # show_progress="hidden"  # CHANGED: Hide the floating progress bar completely
    )
    
    gr.Markdown("""
    ---
    ### Instructions
    
    1. **Setup Portfolio**: 
       - **Manual**: Enter budget and add stocks in the table (Ticker + Quantity)
       - **JSON Upload**: Upload a JSON file with your portfolio
    2. **Choose Strategy**: Select a trading strategy that matches your risk tolerance
    3. **Set Iterations**: Choose how many trading cycles to run
    4. **Start Trading**: Click the button and watch live updates as each iteration completes!
    
    **JSON Format Example:**
    ```json
    {
        "budget": 4000,
        "stocks": {
            "AAPL": 0.56,
            "MSFT": 1.45,
            "GOOGL": 2.0
        }
    }
    ```
    
    💡 **Tip**: The trade history and portfolio comparison update after each iteration in real-time!
    """)

if __name__ == "__main__":
    # app.queue()
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)