"""
Experiment: does the model find a real edge if we give it (a) more data by
training across multiple tickers at once, and (b) a less noisy prediction
target (5-day forward direction instead of 1-day)?

This is deliberately a SEPARATE script from train.py / backtest.py /
walk_forward.py rather than a modification of them, so you can directly
compare "single ticker, 1-day" vs "multi-ticker, 5-day" results side by side.

Design:
- Each ticker is processed independently through indicators + labeling
  (so one ticker's data never leaks into another's feature calculation).
- Each ticker is split chronologically (80/20 by default, same as config.TRAIN_TEST_SPLIT).
- Sequences are built per-ticker (so a sliding window never crosses from one
  ticker's data into another's).
- All tickers' TRAINING sequences are pooled together to train ONE shared
  model -- more data, and it has to learn patterns general enough to work
  across different stocks, not memorize one stock's specific history.
- Each ticker's TEST sequences are evaluated and backtested SEPARATELY, so
  you can see whether the shared model works consistently across stocks or
  only on a lucky few (same spirit as walk_forward.py, but split by ticker
  instead of by time period).

Note: the backtest logic here reuses the same daily buy/sell threshold
strategy as backtest.py even though labels now look 5 days ahead. That's a
simplification -- the model is trained to predict 5-day direction, but the
strategy still checks the signal daily. This is intentional and fine for a
first pass: it tells us whether the *prediction* has any value at all before
we bother building a more realistic 5-day holding-period strategy.
"""

import pickle
import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import config
from data_loader import fetch_data
from features import prepare_dataset, build_sequences
from train import train_model
from backtest import backtest_strategy, buy_and_hold, compute_metrics


def load_and_split_ticker(ticker, horizon):
    """Fetches one ticker, engineers features/labels, and splits chronologically."""
    raw_df = fetch_data(ticker=ticker)
    features, labels, dates, feat_df = prepare_dataset(raw_df, horizon=horizon)

    split_idx = int(len(features) * config.TRAIN_TEST_SPLIT)
    return {
        "ticker": ticker,
        "train_features": features[:split_idx],
        "train_labels": labels[:split_idx],
        "test_features": features[split_idx:],
        "test_labels": labels[split_idx:],
        "test_dates": dates[split_idx:],
        "feat_df": feat_df,
    }


def run_multi_asset_experiment(tickers=None, horizon=None, epochs=None):
    tickers = tickers or config.TICKERS
    horizon = horizon or config.PREDICTION_HORIZON_DAYS
    epochs = epochs or config.EPOCHS
    window = config.LOOKBACK_WINDOW

    print(f"Tickers: {tickers}")
    print(f"Prediction horizon: {horizon} trading day(s)")
    print(f"Lookback window: {window}\n")

    # 1. Load and split each ticker independently
    per_ticker = {}
    for ticker in tickers:
        try:
            per_ticker[ticker] = load_and_split_ticker(ticker, horizon)
        except Exception as e:
            print(f"Skipping {ticker}: {e}")

    if len(per_ticker) < 2:
        raise RuntimeError("Need at least 2 tickers with usable data to run this experiment.")

    # 2. Fit ONE scaler on all tickers' training features combined
    all_train_features = np.concatenate([d["train_features"] for d in per_ticker.values()], axis=0)
    scaler = StandardScaler()
    scaler.fit(all_train_features)

    os.makedirs("models", exist_ok=True)
    with open(config.MULTI_SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    # 3. Build sequences PER TICKER (never crossing ticker boundaries), then pool training sequences
    X_train_list, y_train_list = [], []
    for ticker, d in per_ticker.items():
        train_scaled = scaler.transform(d["train_features"])
        X_tr, y_tr = build_sequences(train_scaled, d["train_labels"], window)
        X_train_list.append(X_tr)
        y_train_list.append(y_tr)
        d["test_scaled"] = scaler.transform(d["test_features"])

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    print(f"Pooled training sequences across {len(per_ticker)} tickers: {X_train.shape}")
    print(f"Pooled train label balance: {y_train.mean():.3f}\n")

    # For early stopping during training we need SOME held-out set -- use a
    # small slice of the pooled training data's tail (chronologically last
    # portion within the pool), not the real test set, to avoid leaking the
    # actual evaluation data into the training loop's stopping decision.
    val_cut = int(len(X_train) * 0.9)
    X_train_fit, y_train_fit = X_train[:val_cut], y_train[:val_cut]
    X_val, y_val = X_train[val_cut:], y_train[val_cut:]

    # 4. Train ONE shared model
    model, _ = train_model(
        X_train_fit, y_train_fit, X_val, y_val,
        input_size=X_train.shape[2], epochs=epochs, verbose=True,
    )
    torch.save(model.state_dict(), config.MULTI_MODEL_PATH)
    print(f"\nShared model saved to {config.MULTI_MODEL_PATH}\n")

    # 5. Evaluate + backtest EACH ticker's test period separately
    results = []
    for ticker, d in per_ticker.items():
        X_test, y_test = build_sequences(d["test_scaled"], d["test_labels"], window)
        if len(X_test) == 0:
            print(f"{ticker}: not enough test data after sequencing, skipping.")
            continue

        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            test_preds = model(X_test_t).numpy()

        test_preds_binary = (test_preds > 0.5).astype(int)
        acc = accuracy_score(y_test, test_preds_binary)
        majority_baseline = max(y_test.mean(), 1 - y_test.mean())

        test_dates = d["test_dates"][-len(y_test):]
        test_prices = d["feat_df"].loc[test_dates, "Close"]

        strategy_values, trade_log = backtest_strategy(test_prices, test_preds, test_dates)
        baseline_values = buy_and_hold(test_prices)

        strategy_metrics = compute_metrics(strategy_values, f"{ticker} strategy", quiet=True)
        baseline_metrics = compute_metrics(baseline_values, f"{ticker} buy&hold", quiet=True)

        print(f"--- {ticker} ---")
        print(f"  Test period: {test_dates[0].date()} to {test_dates[-1].date()}")
        print(f"  Model accuracy: {acc:.4f}  |  Majority baseline: {majority_baseline:.4f}  |  "
              f"{'beats' if acc > majority_baseline else 'does NOT beat'} baseline")
        print(f"  Strategy return: {strategy_metrics['total_return']*100:6.2f}%  |  "
              f"Buy&Hold return: {baseline_metrics['total_return']*100:6.2f}%  |  Trades: {len(trade_log)}\n")

        results.append({
            "ticker": ticker,
            "model_accuracy": acc,
            "majority_baseline": majority_baseline,
            "beats_accuracy_baseline": acc > majority_baseline,
            "strategy_return": strategy_metrics["total_return"],
            "buyhold_return": baseline_metrics["total_return"],
            "beats_buyhold": strategy_metrics["total_return"] > baseline_metrics["total_return"],
            "num_trades": len(trade_log),
        })

    results_df = pd.DataFrame(results)
    print("=" * 60)
    print("=== Multi-Ticker Summary ===\n")
    print(results_df.to_string(index=False))

    n_beat_acc = results_df["beats_accuracy_baseline"].sum()
    n_beat_bh = results_df["beats_buyhold"].sum()
    n_total = len(results_df)

    print(f"\nTickers where model beat the accuracy baseline: {n_beat_acc}/{n_total}")
    print(f"Tickers where strategy beat buy-and-hold:        {n_beat_bh}/{n_total}")
    print(f"Average strategy return:                          {results_df['strategy_return'].mean()*100:.2f}%")
    print(f"Average buy-and-hold return:                      {results_df['buyhold_return'].mean()*100:.2f}%")

    print("\nHow to read this:")
    print(f"- With {horizon}-day horizon + {n_total} tickers pooled, does the model beat baseline")
    print("  on a clear MAJORITY of tickers? If yes, this is much more promising than the")
    print("  single-ticker 1-day result. If it's still ~1 in 5, that's strong evidence this")
    print("  set of features/architecture doesn't have a real directional edge here --")
    print("  time to consider pivoting the project goal (e.g. volatility prediction).")

    return results_df


if __name__ == "__main__":
    run_multi_asset_experiment()