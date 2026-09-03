"""
Central configuration for the AI trading bot.
Edit these values to change ticker, date range, and model settings.
"""

# --- Data settings ---
TICKER = "AAPL"              # Stock to trade (single ticker to start)
START_DATE = "2015-01-01"
END_DATE = None              # None = up to today
INTERVAL = "1d"              # "1d" daily bars (simplest, most reliable with yfinance)

# --- Feature/window settings ---
LOOKBACK_WINDOW = 30         # How many past days the model sees to make one prediction
TRAIN_TEST_SPLIT = 0.8       # 80% train, 20% test (chronological split, NOT random!)

# --- Model settings ---
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4          # L2 regularization -- penalizes overly large weights, reduces overfitting
BATCH_SIZE = 32
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10  # Stop if test loss hasn't improved in this many epochs

# --- Backtest settings ---
INITIAL_CAPITAL = 10_000
TRANSACTION_COST_PCT = 0.001  # 0.1% per trade, to simulate real-world friction
CONFIDENCE_THRESHOLD = 0.52   # Only trade when model confidence exceeds this

# --- Paths ---
DATA_DIR = "data"
MODEL_PATH = "models/lstm_model.pt"
SCALER_PATH = "models/scaler.pkl"

# --- Multi-ticker / longer-horizon experiment settings ---
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "V"]  # diversified large caps
PREDICTION_HORIZON_DAYS = 5   # predict direction over 5 trading days instead of 1 (less noisy)
MULTI_MODEL_PATH = "models/lstm_model_multi.pt"
MULTI_SCALER_PATH = "models/scaler_multi.pkl"

# --- Alpaca paper trading settings ---
# NEVER hardcode real API keys here. Set these as environment variables instead:
#   export ALPACA_API_KEY="your_key_here"
#   export ALPACA_SECRET_KEY="your_secret_here"
import os
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
ALPACA_PAPER = True   # True = paper trading (fake money). Do not flip to False lightly.

# Risk management for live/paper execution -- these matter more than the model itself
MAX_POSITION_PCT = 0.25      # Never put more than 25% of account equity in one position
STOP_LOSS_PCT = 0.05         # Exit a position if it drops 5% from entry, regardless of model signal
MAX_DAILY_TRADES = 3         # Circuit breaker: stop trading for the day past this many trades