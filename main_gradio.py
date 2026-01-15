import gradio as gr
import json
import pandas as pd
from pathlib import Path
import tempfile
from trader_gr import TradingAgent
import requests

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
                "Total": f"${trade['total']:.2f}"
            })
    
    if not all_trades:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_trades)
    return df

def format_portfolio_comparison(initial, final, initial_stock_prices, final_stock_prices):
    """Format initial vs final portfolio comparison with total value including price changes."""
    # Calculate initial total value (using initial prices)
    initial_cash = initial.get('budget', 0)
    initial_stocks_value = 0
    
    for ticker, qty in initial.get('stocks', {}).items():
        price = initial_stock_prices.get(ticker, {}).get('Close', 0)
        initial_stocks_value += qty * price
    
    initial_total = initial_cash + initial_stocks_value
    
    # Calculate final total value (using final/current prices)
    final_cash = final.get('budget', 0)
    final_stocks_value = 0
    
    for ticker, qty in final.get('stocks', {}).items():
        price = final_stock_prices.get(ticker, {}).get('Close', 0)
        final_stocks_value += qty * price
    
    final_total = final_cash + final_stocks_value
    
    # Calculate baseline (no-trade) portfolio value
    # What would the portfolio be worth if we held initial positions at final prices?
    baseline_cash = initial_cash
    baseline_stocks_value = 0
    
    for ticker, qty in initial.get('stocks', {}).items():
        final_price = final_stock_prices.get(ticker, {}).get('Close', 0)
        baseline_stocks_value += qty * final_price
    
    baseline_total = baseline_cash + baseline_stocks_value
    
    # Calculate changes
    total_change = final_total - initial_total
    total_change_pct = ((total_change / initial_total) * 100) if initial_total > 0 else 0
    
    # Calculate trading performance vs baseline
    baseline_change = baseline_total - initial_total
    baseline_change_pct = ((baseline_change / initial_total) * 100) if initial_total > 0 else 0
    
    trading_alpha = final_total - baseline_total  # How much better/worse than doing nothing
    trading_alpha_pct = ((trading_alpha / baseline_total) * 100) if baseline_total > 0 else 0
    
    comparison = "### Portfolio Comparison\n\n"
    comparison += "#### Cash & Assets\n\n"
    comparison += f"**Initial Cash:** ${initial_cash:,.2f}\n"
    comparison += f"**Initial Stocks Value:** ${initial_stocks_value:,.2f}\n"
    comparison += f"**Initial Total Value:** ${initial_total:,.2f}\n\n"
    
    comparison += f"**Final Cash:** ${final_cash:,.2f}\n"
    comparison += f"**Final Stocks Value:** ${final_stocks_value:,.2f}\n"
    comparison += f"**Final Total Value:** ${final_total:,.2f}\n\n"
    
    change_symbol = "📈" if total_change >= 0 else "📉"
    comparison += f"{change_symbol} **Total Change:** ${total_change:,.2f} ({total_change_pct:+.2f}%)\n\n"
    
    # Add baseline comparison
    comparison += "#### 📊 No-Trade Baseline (Hold Strategy)\n\n"
    comparison += f"**Baseline Portfolio Value:** ${baseline_total:,.2f}\n"
    comparison += f"**Baseline Change:** ${baseline_change:,.2f} ({baseline_change_pct:+.2f}%)\n\n"
    
    # Trading performance vs baseline
    alpha_symbol = "✅" if trading_alpha >= 0 else "❌"
    comparison += f"{alpha_symbol} **Trading Alpha (vs Hold):** ${trading_alpha:,.2f} ({trading_alpha_pct:+.2f}%)\n"
    
    if trading_alpha >= 0:
        comparison += f"*Your trading strategy outperformed a simple hold strategy!*\n\n"
    else:
        comparison += f"*A hold strategy would have performed better.*\n\n"
    
    comparison += "---\n\n"
    
    comparison += "#### Stock Holdings\n\n"
    comparison += "| Ticker | Initial Qty | Final Qty | Initial Price | Final Price | Value Change |\n"
    comparison += "|--------|-------------|-----------|---------------|-------------|-------------|\n"
    
    all_tickers = set(list(initial.get('stocks', {}).keys()) + list(final.get('stocks', {}).keys()))
    
    for ticker in sorted(all_tickers):
        init_qty = initial.get('stocks', {}).get(ticker, 0)
        final_qty = final.get('stocks', {}).get(ticker, 0)
        
        init_price = initial_stock_prices.get(ticker, {}).get('Close', 0)
        final_price = final_stock_prices.get(ticker, {}).get('Close', 0)
        
        # Calculate value change: (final_qty * final_price) - (init_qty * init_price)
        init_value = init_qty * init_price
        final_value = final_qty * final_price
        value_change = final_value - init_value
        value_change_str = f"+${value_change:.2f}" if value_change >= 0 else f"-${abs(value_change):,.2f}"
        
        comparison += f"| {ticker} | {init_qty} | {final_qty} | ${init_price:.2f} | ${final_price:.2f} | {value_change_str} |\n"
    
    return comparison

def run_trading(budget, stocks_df, json_file, strategy, num_iterations):
    """
    Main function to run the trading agent with live updates.
    """
    try:
        # Progress tracking with loading state - hide table initially
        yield gr.update(visible=False), "", "⏳ Setting up portfolio..."
        
        # Determine portfolio source
        if json_file is not None:
            portfolio = agent.load_portfolio_from_file(json_file)
            budget = portfolio.get("budget", 0)
            stocks = portfolio.get("stocks", {})
        else:
            if not budget or budget <= 0:
                yield gr.update(visible=False), "", "❌ Error: Please provide a valid budget."
                return
            
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
                yield gr.update(visible=False), "", "❌ Error: Please provide at least one stock holding."
                return
            
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            temp_file.close()
            agent.setup_portfolio(budget, stocks, temp_file.name)
        
        yield gr.update(visible=False), """<div align="center">

<style>
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
.spinner {
    border: 4px solid #f3f3f3;
    border-top: 4px solid #3498db;
    border-radius: 50%;
    width: 40px;
    height: 40px;
    animation: spin 1s linear infinite;
    margin: 20px auto;
}
</style>

<h3>🤖 AI Agent Initializing</h3>
<p>
📊 Analyzing market conditions...<br>
📰 Gathering latest news and sentiment data...<br>
🔍 Evaluating portfolio holdings...<br>
⚙️ Preparing trading strategy...<br>
⏳ Starting first iteration...
</p>


<img src="https://media1.tenor.com/m/4y21cGULStEAAAAd/meme-crypto.gif" width="400" alt="Loading..."/>



</div>""", f"⏳ Starting {num_iterations} iteration(s)..."
        

        # <div class="spinner"></div>
        # Run iterations with live updates
        all_results = []
        initial_portfolio = None
        initial_stock_prices = None
        initial_summary_shown = False
        
        for i in range(1, num_iterations + 1):
            # Update date file (simulating progression)
            with open("/teamspace/studios/this_studio/TraderAgent/data/current_date.txt", "w") as date_file:
                date_file.write(f"{i+15}\n")
            
            # Store initial portfolio on first iteration
            if initial_portfolio is None:
                with open(agent.portfolio_file, 'r') as file:
                    initial_portfolio = json.load(file)
            
            result = agent.run_iteration(strategy)
            
            result["iteration"] = i
            all_results.append(result)
            
            # Capture initial stock prices from first iteration
            if initial_stock_prices is None:
                initial_stock_prices = agent.stock_current.copy()
            
            # Get final portfolio from result
            final_portfolio = result["final_portfolio"]
            
            # Show initial portfolio value before first iteration
            if not initial_summary_shown:
                initial_cash = initial_portfolio.get('budget', 0)
                initial_stocks_value = sum(
                    initial_portfolio.get('stocks', {}).get(ticker, 0) * initial_stock_prices.get(ticker, {}).get('Close', 0)
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

{format_portfolio_comparison(initial_portfolio, final_portfolio, initial_stock_prices, agent.stock_current)}
"""
            
            progress_text = f"⏳ Iteration {i}/{num_iterations} complete. {total_trades} trade(s) so far..."

            yield gr.update(value=trade_history_df, visible=True), summary, progress_text
        
        # Final update
        trade_history_df = format_trade_history(all_results)
        total_trades = sum(len(r.get("trade_history", [])) for r in all_results)
        
        # Calculate initial total value
        initial_cash = initial_portfolio.get('budget', 0)
        initial_stocks_value = sum(
            initial_portfolio.get('stocks', {}).get(ticker, 0) * initial_stock_prices.get(ticker, {}).get('Close', 0)
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

{format_portfolio_comparison(initial_portfolio, final_portfolio, initial_stock_prices, agent.stock_current)}
"""
        yield trade_history_df, summary, "✅ Trading completed successfully!"
        
    except Exception as e:
        yield gr.update(visible=False), "", f"❌ Trading failed: {str(e)}"

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
        "Ticker": ["AAPL", "MSFT", "GOOGL", "META", "TSLA"],
        "Quantity": [2, 2, 1, 2, 3]
    })

# Create Gradio interface
with gr.Blocks(title="Auto-Trading Agent", theme=gr.themes.Soft()) as app:

    gr.Markdown("""
    <div style="text-align: center;">
    <h1> 🤖 Auto-Trading Agent </h1>
    
    Configure your portfolio and let the AI agent make trading decisions based on stock predictions.
    </div>
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
                    column_count=(2, "fixed"),
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
                    "Low Risk - Long Term Conservative"
                ],
                value="Medium Risk - Balanced",
                info="Select your preferred trading strategy"
            )
            
            iterations_input = gr.Slider(
                label="Number of Iterations",
                minimum=1,
                maximum=20,
                value=5,
                step=1,
                info="How many trading cycles to run"
            )
            
            run_button = gr.Button("🚀 Start Trading", variant="primary", size="lg")
            
            status_output = gr.Markdown("")
        
        with gr.Column(scale=2):
            
            summary_output = gr.Markdown("""
### 🎯 How It Works

**Step 1: Load Your Portfolio** 💼  
Choose your style: manually enter your holdings or drop in a JSON file. Either way, make sure you've got some skin in the game!

**Step 2: Pick Your Strategy** 🎲  
Are you a risk-taker or playing it safe? Choose a strategy that matches your vibe (and risk tolerance).

**Step 3: Set the Pace** ⚡  
Decide how many trading cycles you want to run. More iterations = more opportunities for the AI to work its magic.

**Step 4: Let the AI Cook** 🤖  
Hit that button and watch the agent analyze market trends, sentiment, and news in real-time. Every trade is calculated, every move is strategic.

---

### 📋 JSON Portfolio Format

Want to upload your portfolio? Here's the structure:
```json
{
    "budget": 4000,
    "stocks": {
        "AAPL": 3,
        "MSFT": 2,
        "GOOGL": 4
    }
}
```

💡 **Pro Tip**: Keep an eye on the live updates - you'll see exactly when trades happen, how your portfolio shifts, and whether you're beating the "hold and pray" strategy!

🚀 **Ready to trade smarter? Let's go!**"""
)
            
            trade_history_output = gr.Dataframe(
                label="Trade History",
                headers=["Iteration", "Ticker", "Action", "Quantity", "Price", "Total"],
                wrap=True,
                visible=False  # Start hidden, will show when first iteration completes
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
            status_output
        ]
    )
    

    gr.Markdown("""
    ---
    <div style="text-align: center;">
    <h3> ✨ Crafted by Team Overfit </h3>
    Văcărașu Dragoș-Ștefan · Țarcă Andrei-Ioan · Noje Raul <br>
    <em>Built for educational purposes. Trade responsibly - past performance doesn't guarantee future results.<em>
    </div>
    """
    )

    gr.Markdown(""" 
    ---
    """)

if __name__ == "__main__":
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)