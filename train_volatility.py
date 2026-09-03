"""
Trains the LSTM to predict volatility expansion instead of price direction:
"will realized volatility over the next N days be higher than the trailing
volatility over the past M days?"

This reuses the exact same architecture, features, and training loop as the
direction-prediction model (train.py) -- only the label changes. See
features.make_volatility_labels for the label definition and the reasoning
for why volatility tends to be more learnable than direction.
"""

import os
import pickle

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

import config
from data_loader import fetch_data
from features import prepare_volatility_dataset, build_sequences
from train import train_model


def run_volatility_training(ticker=None, horizon=None, lookback=None):
    ticker = ticker or config.TICKER
    horizon = horizon or config.PREDICTION_HORIZON_DAYS
    lookback = lookback or config.VOLATILITY_LOOKBACK_DAYS

    raw_df = fetch_data(ticker=ticker)
    features, labels, dates, feat_df = prepare_volatility_dataset(raw_df, horizon=horizon, lookback=lookback)

    split_idx = int(len(features) * config.TRAIN_TEST_SPLIT)
    train_features, test_features = features[:split_idx], features[split_idx:]
    train_labels, test_labels = labels[:split_idx], labels[split_idx:]

    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)

    os.makedirs("models", exist_ok=True)
    with open(config.VOL_SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    window = config.LOOKBACK_WINDOW
    X_train, y_train = build_sequences(train_features_scaled, train_labels, window)
    X_test, y_test = build_sequences(test_features_scaled, test_labels, window)

    print(f"Ticker: {ticker}  |  Volatility horizon: {horizon}d  |  Trailing lookback: {lookback}d")
    print(f"Train sequences: {X_train.shape}, Test sequences: {X_test.shape}")
    print(f"Train label balance: {y_train.mean():.3f} (fraction of 'volatility expanding' days)")
    print(f"Test label balance: {y_test.mean():.3f}")

    model, test_preds = train_model(X_train, y_train, X_test, y_test, input_size=X_train.shape[2])

    test_preds_binary = (test_preds > 0.5).astype(int)
    print("\n=== Final test set report (volatility expansion prediction) ===")
    print(classification_report(y_test, test_preds_binary, target_names=["Contract/flat", "Expand"]))

    majority_baseline_acc = max(y_test.mean(), 1 - y_test.mean())
    print(f"Naive 'always predict majority class' baseline accuracy: {majority_baseline_acc:.4f}")
    print(f"Model accuracy: {accuracy_score(y_test, test_preds_binary):.4f}")

    torch.save(model.state_dict(), config.VOL_MODEL_PATH)
    print(f"\nModel saved to {config.VOL_MODEL_PATH}")

    test_dates = dates[-len(y_test):]
    test_prices = feat_df.loc[test_dates, "Close"]

    return model, scaler, (X_test, y_test, test_preds, test_dates, test_prices)


if __name__ == "__main__":
    run_volatility_training()