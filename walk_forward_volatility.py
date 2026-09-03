"""
Walk-forward validation for the volatility-expansion model -- same discipline
as walk_forward.py (expanding-window retraining, multiple sequential
out-of-sample test periods), applied to the volatility model instead of the
direction model. A single good-looking train/test split fooled this project
once already; don't trust the promising single-split volatility result
without this.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import config
from data_loader import fetch_data
from features import prepare_volatility_dataset, build_sequences
from train import train_model
from backtest import compute_metrics
from backtest_volatility import simulate_position_scaling, evaluate_volatility_prediction_quality
from walk_forward import generate_folds


def run_volatility_walk_forward(ticker=None, horizon=None, lookback=None, n_folds=5, epochs=None, verbose_training=False):
    ticker = ticker or config.TICKER
    horizon = horizon or config.PREDICTION_HORIZON_DAYS
    lookback = lookback or config.VOLATILITY_LOOKBACK_DAYS
    epochs = epochs or config.EPOCHS
    window = config.LOOKBACK_WINDOW

    raw_df = fetch_data(ticker=ticker)
    features, labels, dates, feat_df = prepare_volatility_dataset(raw_df, horizon=horizon, lookback=lookback)

    folds = generate_folds(len(features), n_folds=n_folds)
    if not folds:
        raise ValueError("Not enough data for walk-forward validation with current settings.")

    print(f"Volatility walk-forward for {ticker}: {len(folds)} folds, "
          f"{horizon}-day horizon, {lookback}-day trailing lookback\n")

    fold_results = []

    for fold_idx, (train_end, test_end) in enumerate(folds, start=1):
        print(f"{'=' * 60}\nFold {fold_idx}: train on rows [0:{train_end}], test on rows [{train_end}:{test_end}]")
        print(f"  Train period: {dates[0].date()} to {dates[train_end - 1].date()}")
        print(f"  Test period:  {dates[train_end].date()} to {dates[test_end - 1].date()}")

        train_features = features[:train_end]
        train_labels = labels[:train_end]

        context_start = train_end - window
        test_features_ctx = features[context_start:test_end]
        test_labels_ctx = labels[context_start:test_end]

        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        test_features_scaled = scaler.transform(test_features_ctx)

        X_train, y_train = build_sequences(train_features_scaled, train_labels, window)
        X_test, y_test = build_sequences(test_features_scaled, test_labels_ctx, window)

        model, test_preds = train_model(
            X_train, y_train, X_test, y_test,
            input_size=X_train.shape[2], epochs=epochs, verbose=verbose_training,
            seed=42 + fold_idx,
        )

        test_preds_binary = (test_preds > 0.5).astype(int)
        acc = accuracy_score(y_test, test_preds_binary)
        majority_baseline = max(y_test.mean(), 1 - y_test.mean())

        fold_dates = dates[train_end:test_end]
        fold_prices = feat_df.loc[fold_dates, "Close"]

        baseline_values, managed_values, position_sizes = simulate_position_scaling(fold_prices, test_preds)
        baseline_metrics = compute_metrics(baseline_values, f"Fold {fold_idx} baseline", quiet=True)
        managed_metrics = compute_metrics(managed_values, f"Fold {fold_idx} managed", quiet=True)

        drawdown_improved = managed_metrics["max_drawdown"] > baseline_metrics["max_drawdown"]  # closer to 0 = better

        print(f"  Model accuracy: {acc:.4f}  |  Majority baseline: {majority_baseline:.4f}  |  "
              f"{'beats' if acc > majority_baseline else 'does NOT beat'} baseline")
        print(f"  Baseline drawdown: {baseline_metrics['max_drawdown']*100:.2f}%  |  "
              f"Managed drawdown: {managed_metrics['max_drawdown']*100:.2f}%  |  "
              f"{'improved' if drawdown_improved else 'did NOT improve'}")
        print(f"  Return given up: {(baseline_metrics['total_return'] - managed_metrics['total_return'])*100:+.2f}pp  |  "
              f"Time at reduced size: {(position_sizes < 1.0).mean()*100:.1f}%")

        evaluate_volatility_prediction_quality(fold_prices, test_preds, horizon, lookback)

        fold_results.append({
            "fold": fold_idx,
            "test_start": dates[train_end],
            "test_end": dates[test_end - 1],
            "model_accuracy": acc,
            "majority_baseline_accuracy": majority_baseline,
            "beats_accuracy_baseline": acc > majority_baseline,
            "baseline_drawdown": baseline_metrics["max_drawdown"],
            "managed_drawdown": managed_metrics["max_drawdown"],
            "drawdown_improved": drawdown_improved,
            "return_given_up": baseline_metrics["total_return"] - managed_metrics["total_return"],
        })
        print()

    results_df = pd.DataFrame(fold_results)

    print(f"{'=' * 60}\n=== Volatility Walk-Forward Summary ({len(folds)} folds) ===\n")
    print(results_df.to_string(index=False))

    n_beat_acc = results_df["beats_accuracy_baseline"].sum()
    n_dd_improved = results_df["drawdown_improved"].sum()
    avg_return_given_up = results_df["return_given_up"].mean()

    print(f"\nFolds where model beat the accuracy baseline: {n_beat_acc}/{len(folds)}")
    print(f"Folds where drawdown improved:                  {n_dd_improved}/{len(folds)}")
    print(f"Average return given up by managing risk:       {avg_return_given_up*100:+.2f}pp")

    print("\nHow to read this:")
    print("- Look for the accuracy baseline AND drawdown improvement to hold up in a")
    print("  clear majority of folds, consistently -- not just the single split you saw")
    print("  before. One fold with a big drawdown improvement and others with none is")
    print("  the same overfitting/regime-dependence warning sign as with direction.")

    return results_df


if __name__ == "__main__":
    run_volatility_walk_forward()