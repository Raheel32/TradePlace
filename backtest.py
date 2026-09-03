"""
Simulates trading the model's predictions on the held-out test set,
including transaction costs, and compares against a buy-and-hold baseline.

This is the step that tells you whether a model with decent classification
accuracy actually translates into money -- these are NOT the same thing.
"""

import numpy as np
import pandas as pd

import config
from train import run_training


def backtest_strategy(prices: pd.Series, predictions: np.ndarray, dates,
                       initial_capital=None, confidence_threshold=None,
                       transaction_cost_pct=None):
    """
    prices: the actual close prices aligned with the prediction dates
    predictions: model's predicted probability of "up" for each day
    """
    initial_capital = initial_capital or config.INITIAL_CAPITAL
    confidence_threshold = confidence_threshold or config.CONFIDENCE_THRESHOLD
    transaction_cost_pct = transaction_cost_pct or config.TRANSACTION_COST_PCT

    cash = initial_capital
    shares = 0
    position = "flat"  # "flat" or "long"
    portfolio_values = []
    trade_log = []

    for i in range(len(predictions) - 1):
        price_today = prices.iloc[i]
        price_tomorrow = prices.iloc[i + 1]
        prob_up = predictions[i]

        # Simple strategy: go long if confident it'll rise, exit if confident it'll fall
        if prob_up > confidence_threshold and position == "flat":
            shares = (cash * (1 - transaction_cost_pct)) / price_today
            cash = 0
            position = "long"
            trade_log.append((dates[i], "BUY", price_today))

        elif prob_up < (1 - confidence_threshold) and position == "long":
            cash = shares * price_today * (1 - transaction_cost_pct)
            shares = 0
            position = "flat"
            trade_log.append((dates[i], "SELL", price_today))

        portfolio_value = cash + shares * price_today
        portfolio_values.append(portfolio_value)

    # Liquidate at the end if still holding
    final_price = prices.iloc[len(predictions) - 1]
    final_value = cash + shares * final_price
    portfolio_values.append(final_value)

    return np.array(portfolio_values), trade_log


def buy_and_hold(prices: pd.Series, initial_capital=None):
    initial_capital = initial_capital or config.INITIAL_CAPITAL
    shares = initial_capital / prices.iloc[0]
    return shares * prices.values


def compute_metrics(portfolio_values: np.ndarray, label: str, quiet: bool = False):
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
    sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252)  # annualized, daily bars
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - running_max) / running_max
    max_drawdown = drawdown.min()

    if not quiet:
        print(f"\n--- {label} ---")
        print(f"Final value:     ${portfolio_values[-1]:,.2f}")
        print(f"Total return:    {total_return * 100:.2f}%")
        print(f"Sharpe ratio:    {sharpe:.2f}")
        print(f"Max drawdown:    {max_drawdown * 100:.2f}%")

    return {
        "final_value": portfolio_values[-1],
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def run_backtest():
    model, scaler, (X_test, y_test, test_preds, test_dates) = run_training()

    # Need the actual close prices for the test period to simulate P&L
    from data_loader import fetch_data
    from features import add_technical_indicators

    raw_df = fetch_data()
    df = add_technical_indicators(raw_df)
    test_prices = df.loc[test_dates, "Close"]

    strategy_values, trade_log = backtest_strategy(test_prices, test_preds, test_dates)
    baseline_values = buy_and_hold(test_prices)

    strategy_metrics = compute_metrics(strategy_values, "LSTM Strategy")
    baseline_metrics = compute_metrics(baseline_values, "Buy & Hold Baseline")

    print(f"\nNumber of trades executed: {len(trade_log)}")
    if trade_log:
        print("First few trades:")
        for date, action, price in trade_log[:5]:
            print(f"  {date.date()} | {action:4s} @ ${price:.2f}")

    if strategy_metrics["total_return"] > baseline_metrics["total_return"]:
        print("\n✅ Strategy beat buy-and-hold on this test period.")
    else:
        print("\n⚠️  Strategy did NOT beat buy-and-hold on this test period.")
    print("(One test period is not proof of anything -- re-test across different date ranges and tickers before trusting this.)")

    return strategy_values, baseline_values, trade_log


if __name__ == "__main__":
    run_backtest()