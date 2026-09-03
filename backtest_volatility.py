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


def simulate_constant_exposure(prices: pd.Series, avg_exposure: float):
    """
    Fair control for testing whether the model's DAY-BY-DAY SELECTION of when
    to reduce exposure adds value, versus just holding a constant reduced
    position size every day. If the managed strategy (which selectively
    reduces on predicted-high-vol days) doesn't beat this constant-exposure
    control on drawdown, then the earlier "always full position" comparison
    was misleading -- the drawdown improvement would just be a mechanical
    consequence of being under-invested on average, not real timing skill.
    """
    daily_returns = prices.pct_change().dropna().values
    values = config.INITIAL_CAPITAL * np.cumprod(1 + daily_returns * avg_exposure)
    values = np.concatenate([[config.INITIAL_CAPITAL], values])
    return values


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


def evaluate_volatility_prediction_quality(prices: pd.Series, predicted_prob_expand: np.ndarray, horizon: int, lookback: int):
    """
    Checks the signal against the exact quantity the label is defined on:
    the CHANGE in volatility (future horizon-day realized vol minus trailing
    lookback-day vol), not the absolute level. An earlier version of this
    check compared absolute volatility levels between predicted groups,
    which doesn't match the relative "expansion vs. trailing" label
    definition and could show a misleading result even for a model that's
    correctly predicting relative changes.
    """
    daily_returns = prices.pct_change()
    trailing_vol = daily_returns.rolling(lookback).std()
    future_vol = daily_returns.rolling(horizon).std().shift(-horizon)
    delta = (future_vol - trailing_vol).values

    n = min(len(delta), len(predicted_prob_expand))
    delta = delta[:n]
    signals = predicted_prob_expand[:n]

    valid = ~np.isnan(delta)
    delta = delta[valid]
    signals = signals[valid]

    high_vol_predicted = delta[signals > 0.5]
    low_vol_predicted = delta[signals <= 0.5]

    print(f"\n--- Does the signal track actual volatility CHANGE (vs. trailing baseline)? ---")
    if len(high_vol_predicted) > 0:
        print(f"Avg (future vol - trailing vol) on days predicted EXPAND: {high_vol_predicted.mean()*100:+.3f}pp  "
              f"({len(high_vol_predicted)} days)")
    if len(low_vol_predicted) > 0:
        print(f"Avg (future vol - trailing vol) on days predicted CONTRACT/FLAT: {low_vol_predicted.mean()*100:+.3f}pp  "
              f"({len(low_vol_predicted)} days)")
    if len(high_vol_predicted) > 0 and len(low_vol_predicted) > 0:
        if high_vol_predicted.mean() > low_vol_predicted.mean():
            print("-> Predicted-EXPAND days DID show a bigger real increase in volatility. Good sign.")
        else:
            print("-> Predicted-EXPAND days did NOT show a bigger real increase. The signal isn't tracking real risk.")


def run_volatility_backtest(ticker=None, horizon=None, lookback=None):
    horizon = horizon or config.PREDICTION_HORIZON_DAYS
    lookback = lookback or config.VOLATILITY_LOOKBACK_DAYS
    model, scaler, (X_test, y_test, test_preds, test_dates, test_prices) = run_volatility_training(
        ticker=ticker, horizon=horizon, lookback=lookback
    )

    baseline_values, managed_values, position_sizes = simulate_position_scaling(test_prices, test_preds)

    print(f"\n{'='*60}")
    baseline_metrics = compute_metrics(baseline_values, "Always Full Position (baseline)")
    managed_metrics = compute_metrics(managed_values, "Volatility-Managed Position")

    # Fair control: same average exposure as the managed strategy, but applied
    # uniformly every day rather than selectively on predicted-high-vol days
    avg_exposure = position_sizes.mean()
    constant_values = simulate_constant_exposure(test_prices, avg_exposure)
    constant_metrics = compute_metrics(constant_values, f"Constant {avg_exposure*100:.0f}% Exposure (fair control)")

    print(f"\n--- Risk comparison ---")
    vol_reduction = 1 - (managed_metrics['max_drawdown'] / baseline_metrics['max_drawdown']) if baseline_metrics['max_drawdown'] != 0 else 0
    return_given_up = baseline_metrics['total_return'] - managed_metrics['total_return']

    print(f"Max drawdown improved from {baseline_metrics['max_drawdown']*100:.2f}% to {managed_metrics['max_drawdown']*100:.2f}% "
          f"({vol_reduction*100:+.1f}% relative change)")
    print(f"Return given up by managing risk: {return_given_up*100:.2f} percentage points")
    print(f"Time spent at reduced position size: {(position_sizes < 1.0).mean()*100:.1f}% of days")

    print(f"\n--- Is this actual TIMING skill, or just lower average exposure? ---")
    print(f"Average exposure held by the managed strategy: {avg_exposure*100:.1f}%")
    print(f"Managed (selective) drawdown:                  {managed_metrics['max_drawdown']*100:.2f}%")
    print(f"Constant {avg_exposure*100:.0f}% exposure (uniform) drawdown:      {constant_metrics['max_drawdown']*100:.2f}%")
    if managed_metrics['max_drawdown'] > constant_metrics['max_drawdown']:
        print("-> Managed strategy has a BETTER (smaller) drawdown than a naive constant-exposure")
        print("   control at the same average exposure. This is evidence of real timing skill,")
        print("   not just being under-invested on average.")
    else:
        print("-> Managed strategy is NOT better than simply holding a constant reduced position")
        print("   at the same average exposure. The apparent drawdown improvement is likely just")
        print("   a mechanical effect of lower average exposure, not real timing skill.")

    evaluate_volatility_prediction_quality(test_prices, test_preds, horizon, lookback)

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