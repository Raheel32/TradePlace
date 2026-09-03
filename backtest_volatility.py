"""
Translates the volatility model's predictions into a concrete risk-management
action and measures whether it actually helps: on days the model predicts
volatility will expand, scale the position down (config.VOL_POSITION_SCALE_DOWN);
otherwise hold a full position. Compare against a naive always-full-position
baseline.

Note what this is NOT: it is not a directional strategy. Both the "managed"
and "baseline" portfolios are long the same asset the whole time and earn the
same daily returns per dollar invested -- the only difference is how much is
invested on any given day. This isolates the value of volatility prediction
specifically, independent of whether direction is predictable at all.

What "success" looks like here is different from the direction-prediction
backtest: we are NOT trying to beat buy-and-hold's total return (position
sizing down inherently gives up some upside). We're checking whether it
reduces volatility and drawdown by MORE than it gives up in return -- i.e.
whether it improves risk-adjusted outcomes.
"""

import numpy as np
import pandas as pd

import config
from train_volatility import run_volatility_training
from backtest import compute_metrics


def simulate_position_scaling(prices: pd.Series, predicted_prob_expand: np.ndarray, scale_down: float = None):
    """
    Simulates two portfolios over the test period:
      - baseline: always fully invested (100% position)
      - managed: scaled down to `scale_down` on days the model predicts
        volatility will expand, full position otherwise

    Both portfolios experience the SAME asset returns -- only position size differs.
    """
    scale_down = scale_down if scale_down is not None else config.VOL_POSITION_SCALE_DOWN

    daily_returns = prices.pct_change().dropna().values
    # Align predictions with the returns they apply to (prediction at day i
    # informs the position held going into day i+1's return)
    n = min(len(daily_returns), len(predicted_prob_expand) - 1)
    daily_returns = daily_returns[:n]
    signals = predicted_prob_expand[:n]

    position_sizes = np.where(signals > 0.5, scale_down, 1.0)

    baseline_returns = daily_returns                      # always 100% position
    managed_returns = daily_returns * position_sizes       # scaled by signal

    baseline_values = config.INITIAL_CAPITAL * np.cumprod(1 + baseline_returns)
    managed_values = config.INITIAL_CAPITAL * np.cumprod(1 + managed_returns)

    baseline_values = np.concatenate([[config.INITIAL_CAPITAL], baseline_values])
    managed_values = np.concatenate([[config.INITIAL_CAPITAL], managed_values])

    return baseline_values, managed_values, position_sizes


def evaluate_volatility_prediction_quality(prices: pd.Series, predicted_prob_expand: np.ndarray, horizon: int):
    """
    Checks the signal against the SAME horizon it was trained to predict:
    forward `horizon`-day realized volatility, not next single-day move size.
    (An earlier version of this check compared against next-day move size,
    which was the wrong horizon and produced a misleading result.)
    """
    daily_returns = prices.pct_change()
    forward_vol = daily_returns.rolling(horizon).std().shift(-horizon).values

    n = min(len(forward_vol), len(predicted_prob_expand))
    forward_vol = forward_vol[:n]
    signals = predicted_prob_expand[:n]

    valid = ~np.isnan(forward_vol)
    forward_vol = forward_vol[valid]
    signals = signals[valid]

    high_vol_predicted = forward_vol[signals > 0.5]
    low_vol_predicted = signals <= 0.5
    low_vol_predicted = forward_vol[low_vol_predicted]

    print(f"\n--- Does the signal actually track real {horizon}-day forward volatility? ---")
    if len(high_vol_predicted) > 0:
        print(f"Avg realized {horizon}-day volatility on days predicted HIGH vol: {high_vol_predicted.mean()*100:.3f}%  "
              f"({len(high_vol_predicted)} days)")
    if len(low_vol_predicted) > 0:
        print(f"Avg realized {horizon}-day volatility on days predicted LOW vol:  {low_vol_predicted.mean()*100:.3f}%  "
              f"({len(low_vol_predicted)} days)")
    if len(high_vol_predicted) > 0 and len(low_vol_predicted) > 0:
        if high_vol_predicted.mean() > low_vol_predicted.mean():
            print("-> Predicted-high-vol days DID have higher realized forward volatility. Good sign.")
        else:
            print("-> Predicted-high-vol days did NOT have higher realized forward volatility. The signal isn't tracking real risk.")


def run_volatility_backtest(ticker=None, horizon=None):
    horizon = horizon or config.PREDICTION_HORIZON_DAYS
    model, scaler, (X_test, y_test, test_preds, test_dates, test_prices) = run_volatility_training(ticker=ticker, horizon=horizon)

    baseline_values, managed_values, position_sizes = simulate_position_scaling(test_prices, test_preds)

    print(f"\n{'='*60}")
    baseline_metrics = compute_metrics(baseline_values, "Always Full Position (baseline)")
    managed_metrics = compute_metrics(managed_values, "Volatility-Managed Position")

    print(f"\n--- Risk comparison ---")
    vol_reduction = 1 - (managed_metrics['max_drawdown'] / baseline_metrics['max_drawdown']) if baseline_metrics['max_drawdown'] != 0 else 0
    return_given_up = baseline_metrics['total_return'] - managed_metrics['total_return']

    print(f"Max drawdown improved from {baseline_metrics['max_drawdown']*100:.2f}% to {managed_metrics['max_drawdown']*100:.2f}% "
          f"({vol_reduction*100:+.1f}% relative change)")
    print(f"Return given up by managing risk: {return_given_up*100:.2f} percentage points")
    print(f"Time spent at reduced position size: {(position_sizes < 1.0).mean()*100:.1f}% of days")

    evaluate_volatility_prediction_quality(test_prices, test_preds, horizon)

    print("\nHow to read this:")
    print("- This is NOT trying to beat buy-and-hold on raw return -- giving up some")
    print("  upside is the expected cost of ever reducing position size.")
    print("- The question is whether drawdown/volatility improved by MORE than the")
    print("  return given up, and whether predicted-high-vol days actually had bigger")
    print("  real moves (see the check above). If both hold up, and ideally across")
    print("  multiple tickers, this is a genuinely useful risk-management signal even")
    print("  though it never tells you which direction to trade.")

    return baseline_values, managed_values


if __name__ == "__main__":
    run_volatility_backtest()