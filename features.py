"""
Turns raw OHLCV data into model-ready features and sliding-window sequences.
"""

import numpy as np
import pandas as pd
import ta

import config


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a set of common technical indicators as extra columns."""
    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # Trend / momentum
    df["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd = ta.trend.MACD(close)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["sma_10"] = ta.trend.SMAIndicator(close, window=10).sma_indicator()
    df["sma_30"] = ta.trend.SMAIndicator(close, window=30).sma_indicator()
    df["ema_10"] = ta.trend.EMAIndicator(close, window=10).ema_indicator()

    # Volatility
    bb = ta.volatility.BollingerBands(close, window=20)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["atr"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

    # Volume
    df["volume_sma_10"] = volume.rolling(10).mean()

    # Price-derived features (relative, not absolute, so the model generalizes better)
    df["return_1d"] = close.pct_change(1)
    df["return_5d"] = close.pct_change(5)
    df["high_low_pct"] = (high - low) / close
    df["close_open_pct"] = (close - df["Open"]) / df["Open"]

    df = df.dropna()
    return df


def make_labels(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """
    Binary label: 1 if the close `horizon` trading days ahead is higher than
    today's close, else 0. horizon=1 (default) is the original next-day label;
    a larger horizon (e.g. 5) predicts a less noisy, medium-term direction.
    """
    future_return = df["Close"].shift(-horizon) / df["Close"] - 1
    label = (future_return > 0).astype(int)
    return label


FEATURE_COLUMNS = [
    "return_1d", "return_5d", "high_low_pct", "close_open_pct",
    "rsi_14", "macd", "macd_signal", "sma_10", "sma_30", "ema_10",
    "bb_high", "bb_low", "atr", "volume_sma_10",
]


def build_sequences(features: np.ndarray, labels: np.ndarray, window: int):
    """
    Converts a 2D feature array into overlapping sequences of length `window`
    for the LSTM, each paired with the label for the day right after the window.
    """
    X, y = [], []
    for i in range(len(features) - window):
        X.append(features[i:i + window])
        y.append(labels[i + window])
    return np.array(X), np.array(y)


def prepare_dataset(df: pd.DataFrame, horizon: int = 1):
    """
    Full pipeline: add indicators, build labels, drop the last `horizon` rows
    (no label available for them since they look `horizon` days ahead),
    and return feature/label arrays ready for sequencing.
    """
    df = add_technical_indicators(df)
    labels = make_labels(df, horizon=horizon)

    df = df.iloc[:-horizon]       # last `horizon` rows have no label
    labels = labels.iloc[:-horizon]

    features = df[FEATURE_COLUMNS].values
    labels = labels.values
    dates = df.index

    return features, labels, dates, df