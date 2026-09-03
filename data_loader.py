"""
Fetches historical stock data using yfinance and caches it locally.
"""

import os
import pandas as pd
import yfinance as yf

import config


def fetch_data(ticker: str = None, start: str = None, end: str = None,
                interval: str = None, force_refresh: bool = False) -> pd.DataFrame:
    """
    Downloads OHLCV data for a ticker. Caches to a CSV so repeated runs
    don't re-download every time.
    """
    ticker = ticker or config.TICKER
    start = start or config.START_DATE
    end = end or config.END_DATE
    interval = interval or config.INTERVAL

    os.makedirs(config.DATA_DIR, exist_ok=True)
    cache_path = os.path.join(config.DATA_DIR, f"{ticker}_{interval}.csv")

    if os.path.exists(cache_path) and not force_refresh:
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        print(f"Loaded cached data for {ticker}: {len(df)} rows")
        return df

    print(f"Downloading {ticker} data from Yahoo Finance...")
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol and date range.")

    # yfinance sometimes returns MultiIndex columns for single tickers; flatten if so
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.to_csv(cache_path)
    print(f"Downloaded and cached {len(df)} rows to {cache_path}")
    return df


if __name__ == "__main__":
    df = fetch_data()
    print(df.head())
    print(df.tail())
    print(f"\nDate range: {df.index.min()} to {df.index.max()}")