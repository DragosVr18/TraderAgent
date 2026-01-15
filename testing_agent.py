import json
import tempfile
from pathlib import Path
from trader_gr import TradingAgent
from typing import Dict, List
import pandas as pd
from datetime import datetime


class TradingAgentTester:
    """Test the trading agent's performance across multiple runs."""
    
    def __init__(self, output_dir: str = "test_results"):
        self.agent = TradingAgent()
        self.results = []
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create timestamped run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{timestamp}"
        self.run_dir.mkdir(exist_ok=True)
        
        print(f"📁 Results will be saved to: {self.run_dir}")
    
    def calculate_portfolio_value(self, portfolio: Dict, stock_prices: Dict) -> float:
        """Calculate total portfolio value given stock prices."""
        cash = portfolio.get('budget', 0)
        stocks_value = sum(
            portfolio.get('stocks', {}).get(ticker, 0) * stock_prices.get(ticker, {}).get('Close', 0)
            for ticker in portfolio.get('stocks', {}).keys()
        )
        return cash + stocks_value
    
    def run_test(self, 
                 budget: float, 
                 stocks: Dict[str, float], 
                 strategy: str, 
                 num_iterations: int,
                 run_number: int = 1) -> Dict:
        """
        Run a single test of the trading agent.
        
        Args:
            budget: Initial budget
            stocks: Initial stock holdings {ticker: quantity}
            strategy: Trading strategy to use
            num_iterations: Number of trading iterations
            run_number: Run number for this strategy
            
        Returns:
            Dict with test results
        """
        # Setup portfolio
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        temp_file.close()
        self.agent.setup_portfolio(budget, stocks, temp_file.name)
        
        # Track initial state
        initial_portfolio = {"budget": budget, "stocks": stocks.copy()}
        initial_stock_prices = None
        
        all_trades = []
        iteration_details = []
        
        print(f"\n{'='*60}")
        print(f"Starting test: {strategy} (Run #{run_number})")
        print(f"Iterations: {num_iterations}")
        print(f"Initial Budget: ${budget:,.2f}")
        print(f"Initial Stocks: {stocks}")
        print(f"{'='*60}\n")
        
        # Run iterations
        for i in range(1, num_iterations + 1):
            # Update date file (simulating progression)
            with open("/teamspace/studios/this_studio/TraderAgent/data/current_date.txt", "w") as f:
                f.write(f"{i+15}\n")
            
            # Reset flags
            self.agent.fetch_values = True
            self.agent.fetch_news = True
            self.agent.trade_history = []
            
            # Load portfolio before iteration
            with open(self.agent.portfolio_file, 'r') as f:
                portfolio_before = json.load(f)
            
            # Run iteration
            print(f"Running iteration {i}/{num_iterations}...", end=" ")
            result = self.agent.run_iteration(strategy)
            
            # Capture initial prices on first iteration
            if initial_stock_prices is None:
                initial_stock_prices = self.agent.stock_current.copy()
            
            # Load portfolio after iteration
            with open(self.agent.portfolio_file, 'r') as f:
                portfolio_after = json.load(f)
            
            # Calculate portfolio value for this iteration
            iteration_value = self.calculate_portfolio_value(
                portfolio_after, 
                self.agent.stock_current
            )
            
            # Store iteration details
            iteration_details.append({
                "iteration": i,
                "trades": result["trade_history"],
                "portfolio_before": portfolio_before,
                "portfolio_after": portfolio_after,
                "portfolio_value": iteration_value,
                "stock_prices": self.agent.stock_current.copy()
            })
            
            # Collect trades
            all_trades.extend(result["trade_history"])
            print(f"✓ ({len(result['trade_history'])} trades, Value: ${iteration_value:,.2f})")
        
        # Load final portfolio
        with open(self.agent.portfolio_file, 'r') as f:
            final_portfolio = json.load(f)
        
        # Get final stock prices
        final_stock_prices = self.agent.stock_current.copy()
        
        # Calculate values
        initial_total = self.calculate_portfolio_value(initial_portfolio, initial_stock_prices)
        final_total = self.calculate_portfolio_value(final_portfolio, final_stock_prices)
        
        # Calculate baseline (hold strategy)
        baseline_total = self.calculate_portfolio_value(initial_portfolio, final_stock_prices)
        
        # Calculate metrics
        total_return = final_total - initial_total
        total_return_pct = (total_return / initial_total * 100) if initial_total > 0 else 0
        
        baseline_return = baseline_total - initial_total
        baseline_return_pct = (baseline_return / initial_total * 100) if initial_total > 0 else 0
        
        alpha = final_total - baseline_total
        alpha_pct = (alpha / baseline_total * 100) if baseline_total > 0 else 0
        
        # Clean up
        Path(self.agent.portfolio_file).unlink(missing_ok=True)
        
        test_result = {
            "strategy": strategy,
            "run_number": run_number,
            "iterations": num_iterations,
            "num_trades": len(all_trades),
            "initial_value": initial_total,
            "final_value": final_total,
            "baseline_value": baseline_total,
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "baseline_return": baseline_return,
            "baseline_return_pct": baseline_return_pct,
            "alpha": alpha,
            "alpha_pct": alpha_pct,
            "trades": all_trades,
            "initial_portfolio": initial_portfolio,
            "final_portfolio": final_portfolio,
            "initial_stock_prices": initial_stock_prices,
            "final_stock_prices": final_stock_prices,
            "iteration_details": iteration_details
        }
        
        self.results.append(test_result)
        self._print_test_summary(test_result)
        self._save_individual_test(test_result)
        
        return test_result
    
    def _save_individual_test(self, result: Dict):
        """Save individual test results to JSON and CSV files."""
        # Create safe filename
        strategy_name = result['strategy'].replace(' ', '_').replace('-', '')
        run_num = result['run_number']
        filename_base = f"{strategy_name}_run{run_num}"
        
        # Save full JSON result
        json_path = self.run_dir / f"{filename_base}_full.json"
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        # Save trades to CSV
        if result['trades']:
            trades_df = pd.DataFrame(result['trades'])
            trades_csv = self.run_dir / f"{filename_base}_trades.csv"
            trades_df.to_csv(trades_csv, index=False)
        
        # Save iteration-by-iteration portfolio values
        iteration_data = []
        for detail in result['iteration_details']:
            # Calculate baseline value at this iteration
            baseline_value_iter = self.calculate_portfolio_value(
                result['initial_portfolio'],
                detail['stock_prices']
            )
            # Calculate alpha for this iteration
            alpha_iter = detail['portfolio_value'] - baseline_value_iter
            alpha_pct_iter = (alpha_iter / baseline_value_iter * 100) if baseline_value_iter > 0 else 0
            
            iteration_data.append({
                "Iteration": detail['iteration'],
                "Portfolio Value": detail['portfolio_value'],
                "Baseline Value": baseline_value_iter,
                "Alpha": alpha_iter,
                "Alpha %": alpha_pct_iter,
                "Cash": detail['portfolio_after']['budget'],
                "Num Trades": len(detail['trades'])
            })
        
        if iteration_data:
            iter_df = pd.DataFrame(iteration_data)
            iter_csv = self.run_dir / f"{filename_base}_iterations.csv"
            iter_df.to_csv(iter_csv, index=False)
        
        print(f"💾 Saved: {filename_base}_*.json/csv")
    
    def _print_test_summary(self, result: Dict):
        """Print a summary of test results."""
        print(f"\n{'='*60}")
        print(f"TEST RESULTS: {result['strategy']} (Run #{result['run_number']})")
        print(f"{'='*60}")
        print(f"Total Trades: {result['num_trades']}")
        print(f"\nPortfolio Value:")
        print(f"  Initial:  ${result['initial_value']:,.2f}")
        print(f"  Final:    ${result['final_value']:,.2f}")
        print(f"  Baseline: ${result['baseline_value']:,.2f}")
        
        print(f"\nReturns:")
        print(f"  Trading:  ${result['total_return']:,.2f} ({result['total_return_pct']:+.2f}%)")
        print(f"  Baseline: ${result['baseline_return']:,.2f} ({result['baseline_return_pct']:+.2f}%)")
        
        alpha_symbol = "✅" if result['alpha'] >= 0 else "❌"
        print(f"\nAlpha (vs Hold): {alpha_symbol} ${result['alpha']:,.2f} ({result['alpha_pct']:+.2f}%)")
        
        if result['alpha'] >= 0:
            print("  → Trading strategy OUTPERFORMED hold strategy")
        else:
            print("  → Trading strategy UNDERPERFORMED hold strategy")
        print(f"{'='*60}\n")
    
    def run_multiple_tests(self, 
                          budget: float,
                          stocks: Dict[str, float],
                          strategies: List[str],
                          num_iterations: int,
                          num_runs_per_strategy: int = 1) -> pd.DataFrame:
        """
        Run multiple tests across different strategies.
        
        Args:
            budget: Initial budget
            stocks: Initial stock holdings
            strategies: List of strategies to test
            num_iterations: Number of iterations per test
            num_runs_per_strategy: Number of times to run each strategy
            
        Returns:
            DataFrame with aggregated results
        """
        print(f"\n{'#'*60}")
        print(f"RUNNING COMPREHENSIVE TEST SUITE")
        print(f"{'#'*60}")
        print(f"Strategies to test: {len(strategies)}")
        print(f"Runs per strategy: {num_runs_per_strategy}")
        print(f"Iterations per run: {num_iterations}")
        print(f"Total tests: {len(strategies) * num_runs_per_strategy}")
        print(f"{'#'*60}\n")
        
        for strategy in strategies:
            for run in range(1, num_runs_per_strategy + 1):
                print(f"\n[Run {run}/{num_runs_per_strategy}] Testing: {strategy}")
                self.run_test(budget, stocks, strategy, num_iterations, run_number=run)
        
        # Create summary DataFrame
        summary_data = []
        for r in self.results:
            summary_data.append({
                "Strategy": r["strategy"],
                "Run": r["run_number"],
                "Iterations": r["iterations"],
                "Trades": r["num_trades"],
                "Initial Value": f"${r['initial_value']:,.2f}",
                "Final Value": f"${r['final_value']:,.2f}",
                "Return": f"${r['total_return']:,.2f}",
                "Return %": f"{r['total_return_pct']:+.2f}%",
                "Baseline Return %": f"{r['baseline_return_pct']:+.2f}%",
                "Alpha": f"${r['alpha']:,.2f}",
                "Alpha %": f"{r['alpha_pct']:+.2f}%",
                "Beat Baseline": "Yes" if r['alpha'] >= 0 else "No"
            })
        
        df = pd.DataFrame(summary_data)
        
        # Save summary
        summary_path = self.run_dir / "summary_all_tests.csv"
        df.to_csv(summary_path, index=False)
        print(f"\n💾 Summary saved to: {summary_path}")
        
        return df
    
    def get_strategy_stats(self) -> pd.DataFrame:
        """Get aggregated statistics by strategy."""
        if not self.results:
            return pd.DataFrame()
        
        # Group by strategy
        strategy_groups = {}
        for r in self.results:
            strategy = r["strategy"]
            if strategy not in strategy_groups:
                strategy_groups[strategy] = []
            strategy_groups[strategy].append(r)
        
        # Calculate stats
        stats = []
        for strategy, results in strategy_groups.items():
            avg_alpha = sum(r["alpha"] for r in results) / len(results)
            avg_alpha_pct = sum(r["alpha_pct"] for r in results) / len(results)
            avg_return_pct = sum(r["total_return_pct"] for r in results) / len(results)
            win_rate = sum(1 for r in results if r["alpha"] >= 0) / len(results) * 100
            
            avg_trades = sum(r["num_trades"] for r in results) / len(results)
            
            stats.append({
                "Strategy": strategy,
                "Runs": len(results),
                "Avg Trades": f"{avg_trades:.1f}",
                "Avg Return %": f"{avg_return_pct:.2f}%",
                "Avg Alpha": f"${avg_alpha:,.2f}",
                "Avg Alpha %": f"{avg_alpha_pct:+.2f}%",
                "Win Rate": f"{win_rate:.1f}%"
            })
        
        df = pd.DataFrame(stats)
        
        # Save strategy stats
        stats_path = self.run_dir / "strategy_statistics.csv"
        df.to_csv(stats_path, index=False)
        print(f"💾 Strategy stats saved to: {stats_path}")
        
        return df
    
    def save_test_configuration(self, budget: float, stocks: Dict[str, float], 
                               strategies: List[str], num_iterations: int, 
                               num_runs_per_strategy: int):
        """Save test configuration for reproducibility."""
        config = {
            "test_date": datetime.now().isoformat(),
            "initial_budget": budget,
            "initial_stocks": stocks,
            "strategies": strategies,
            "num_iterations": num_iterations,
            "num_runs_per_strategy": num_runs_per_strategy
        }
        
        config_path = self.run_dir / "test_configuration.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"💾 Configuration saved to: {config_path}")


def main():
    """Example usage of the tester."""
    tester = TradingAgentTester(output_dir="test_results")
    
    # Test configuration
    initial_budget = 10000
    initial_stocks = {
        "AAPL": 2,
        "MSFT": 2,
        "GOOGL": 2,
        "NFLX": 2,
        "ORCL": 2,
        "TSLA": 2,
        "AMZN": 2,
        "NVDA": 2,
        "META": 2,
        "INTC": 2,
        "PLTR": 2,
    }
    
    strategies = [
        "High Risk - Aggressive Growth",
    #    "Medium Risk - Balanced",
    #    "Low Risk - Long Term Conservative",
    ]
    
    num_iterations = 25
    num_runs_per_strategy = 1
    
    # Save configuration
    tester.save_test_configuration(
        initial_budget, 
        initial_stocks, 
        strategies, 
        num_iterations, 
        num_runs_per_strategy
    )
    
    # Run tests
    results_df = tester.run_multiple_tests(
        budget=initial_budget,
        stocks=initial_stocks,
        strategies=strategies,
        num_iterations=num_iterations,
        num_runs_per_strategy=num_runs_per_strategy
    )
    
    # Print results
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("STRATEGY STATISTICS")
    print("="*80)
    stats_df = tester.get_strategy_stats()
    print(stats_df.to_string(index=False))
    
    print(f"\n✅ All results saved to: {tester.run_dir}")


if __name__ == "__main__":
    main()