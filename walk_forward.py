"""
Walk-forward validation.

A single train/test split (what train.py does) tells you almost nothing --
it's one lucky or unlucky slice of history. Walk-forward validation retrains
the model on an expanding window of history and tests it on the following
block of unseen data, repeated across multiple sequential folds. This is
much closer to how the model would actually be used in production (retrain
periodically, trade forward) and makes overfitting much harder to hide.

Fold layout (expanding window):

  Fold 1: train [------------]        test [----]
  Fold 2: train [------------------]  test        [----]
  Fold 3: train [------------------------]  test          [----]
  ...

Each fold's test period is data the model has never seen at training time,
and importantly, later folds' test periods are also never used to train
earlier folds -- there's no leakage across folds either.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import config
from data_loader import fetch_data
from features import prepare_dataset, build_sequences, add_technical_indicators
from train import train_model
from backtest import backtest_strategy, buy_and_hold, compute_metrics


def generate_folds(n_samples, n_folds=5, test_size=None, min_train_size=None):
    """
    Returns a list of (train_end, test_end) index pairs for expanding-window
    walk-forward validation. Training always starts at index 0 (uses ALL
    history up to that point); only the test block moves forward each fold.
    """
    if test_size is None:
        test_size = max(30, n_samples // (n_folds + 2))
    if min_train_size is None:
        min_train_size = test_size * 2

    folds = []
    train_end = min_train_size
    for _ in range(n_folds):
        test_end = min(train_end + test_size, n_samples)
        if test_end - train_end < 10:  # not enough data left for a meaningful test block
            break
        folds.append((train_end, test_end))
        train_end = test_end

    return folds


def run_walk_forward(n_folds=5, epochs=None, verbose_training=False):
    epochs = epochs or config.EPOCHS

    raw_df = fetch_data()
    features, labels, dates, feat_df = prepare_dataset(raw_df)
    window = config.LOOKBACK_WINDOW

    folds = generate_folds(len(features), n_folds=n_folds)
    if not folds:
        raise ValueError("Not enough data to run walk-forward validation with the current settings. "
                          "Try a longer date range in config.py or fewer folds.")

    print(f"Running walk-forward validation with {len(folds)} folds "
          f"(requested {n_folds}; reduced if data was too short)\n")

    fold_results = []

    for fold_idx, (train_end, test_end) in enumerate(folds, start=1):
        print(f"{'=' * 60}\nFold {fold_idx}: train on rows [0:{train_end}], test on rows [{train_end}:{test_end}]")
        print(f"  Train period: {dates[0].date()} to {dates[train_end - 1].date()}")
        print(f"  Test period:  {dates[train_end].date()} to {dates[test_end - 1].date()}")

        train_features = features[:train_end]
        train_labels = labels[:train_end]

        # Include `window` rows of context before the test block so we don't
        # lose the first `window` test days to sequence-building.
        context_start = train_end - window
        test_features_ctx = features[context_start:test_end]
        test_labels_ctx = labels[context_start:test_end]

        # Fit scaler on TRAIN ONLY, apply to both -- no leakage across the fold boundary
        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        test_features_scaled = scaler.transform(test_features_ctx)

        X_train, y_train = build_sequences(train_features_scaled, train_labels, window)
        X_test, y_test = build_sequences(test_features_scaled, test_labels_ctx, window)

        model, test_preds = train_model(
            X_train, y_train, X_test, y_test,
            input_size=X_train.shape[2],
            epochs=epochs,
            verbose=verbose_training,
            seed=42 + fold_idx,  # vary seed per fold so results aren't identical by fluke
        )

        test_preds_binary = (test_preds > 0.5).astype(int)
        acc = accuracy_score(y_test, test_preds_binary)
        majority_baseline = max(y_test.mean(), 1 - y_test.mean())

        # Raw probability distribution -- this is what actually decides whether
        # the strategy trades at all, independent of the model's classification
        # accuracy. A model stuck near 0.5 with low variance will barely ever
        # cross CONFIDENCE_THRESHOLD, producing near-zero trades regardless of
        # whether its predictions have any skill.
        prob_mean = test_preds.mean()
        prob_std = test_preds.std()
        prob_min = test_preds.min()
        prob_max = test_preds.max()
        pct_would_buy = (test_preds > config.CONFIDENCE_THRESHOLD).mean()
        pct_would_sell = (test_preds < (1 - config.CONFIDENCE_THRESHOLD)).mean()

        # Backtest this fold's test period
        fold_dates = dates[train_end:test_end]
        fold_prices = feat_df.loc[fold_dates, "Close"]
        strategy_values, trade_log = backtest_strategy(fold_prices, test_preds, fold_dates)
        baseline_values = buy_and_hold(fold_prices)

        strategy_metrics = compute_metrics(strategy_values, f"Fold {fold_idx} Strategy", quiet=True)
        baseline_metrics = compute_metrics(baseline_values, f"Fold {fold_idx} Buy&Hold", quiet=True)

        print(f"  Model accuracy: {acc:.4f}  |  Majority baseline: {majority_baseline:.4f}  |  "
              f"{'beats' if acc > majority_baseline else 'does NOT beat'} baseline")
        print(f"  Predicted probability distribution: mean={prob_mean:.3f}  std={prob_std:.3f}  "
              f"range=[{prob_min:.3f}, {prob_max:.3f}]")
        print(f"  Days that would trigger BUY (>{config.CONFIDENCE_THRESHOLD}): {pct_would_buy*100:.1f}%  |  "
              f"Days that would trigger SELL (<{1-config.CONFIDENCE_THRESHOLD:.2f}): {pct_would_sell*100:.1f}%")
        print(f"  Strategy return: {strategy_metrics['total_return']*100:6.2f}%  |  "
              f"Buy&Hold return: {baseline_metrics['total_return']*100:6.2f}%  |  "
              f"Trades: {len(trade_log)}")

        fold_results.append({
            "fold": fold_idx,
            "test_start": dates[train_end],
            "test_end": dates[test_end - 1],
            "model_accuracy": acc,
            "majority_baseline_accuracy": majority_baseline,
            "beats_accuracy_baseline": acc > majority_baseline,
            "prob_mean": prob_mean,
            "prob_std": prob_std,
            "strategy_return": strategy_metrics["total_return"],
            "strategy_sharpe": strategy_metrics["sharpe"],
            "strategy_max_drawdown": strategy_metrics["max_drawdown"],
            "buyhold_return": baseline_metrics["total_return"],
            "beats_buyhold": strategy_metrics["total_return"] > baseline_metrics["total_return"],
            "num_trades": len(trade_log),
        })

    results_df = pd.DataFrame(fold_results)

    print(f"\n{'=' * 60}\n=== Walk-Forward Summary ({len(folds)} folds) ===\n")
    print(results_df.to_string(index=False))

    n_beat_accuracy = results_df["beats_accuracy_baseline"].sum()
    n_beat_buyhold = results_df["beats_buyhold"].sum()
    avg_strategy_return = results_df["strategy_return"].mean()
    avg_buyhold_return = results_df["buyhold_return"].mean()
    avg_sharpe = results_df["strategy_sharpe"].mean()

    print(f"\nFolds where model beat the accuracy baseline: {n_beat_accuracy}/{len(folds)}")
    print(f"Folds where strategy beat buy-and-hold:        {n_beat_buyhold}/{len(folds)}")
    print(f"Average strategy return per fold:               {avg_strategy_return*100:.2f}%")
    print(f"Average buy-and-hold return per fold:           {avg_buyhold_return*100:.2f}%")
    print(f"Average Sharpe ratio per fold:                  {avg_sharpe:.2f}")

    print("\nHow to read this:")
    print("- A model with a real edge should beat the accuracy baseline in a clear")
    print("  majority of folds, not just 1 out of 5 -- that's what a single lucky")
    print("  split can hide and walk-forward reveals.")
    print("- If results are inconsistent across folds (winning big in one, losing")
    print("  big in another), that's a sign of overfitting or regime-dependence,")
    print("  not a stable edge.")

    return results_df


if __name__ == "__main__":
    run_walk_forward(n_folds=5)