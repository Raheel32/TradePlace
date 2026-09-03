# AI Trading Bot (LSTM, Stocks)

An educational pipeline for building a deep-learning stock prediction + backtesting
system in Python. **This is a learning project, not a proven money-making system.**
Predicting short-term stock direction is genuinely hard, and most simple models do
not reliably beat buy-and-hold once transaction costs are included.

## Project structure

```
trading_bot/
├── config.py              # All settings in one place (ticker, dates, model hyperparams)
├── data_loader.py         # Downloads & caches historical OHLCV data (yfinance)
├── generate_sample_data.py# TEST ONLY: synthetic data, for environments without internet
├── features.py            # Technical indicators + sliding-window sequence builder
├── model.py                # PyTorch LSTM classifier definition
├── train.py                # Trains the model, chronological train/test split
├── backtest.py             # Simulates trading on predictions, compares to buy & hold
├── walk_forward.py         # Multi-fold walk-forward validation (the real test)
├── multi_asset_experiment.py # Trains one shared model across multiple tickers, 5-day horizon
├── alpaca_execution.py     # Connects trained model to Alpaca PAPER trading
├── data/                   # Cached CSV data (gitignore this)
└── models/                 # Saved model weights + scaler (gitignore this)
```

## Setup (on your own machine, with internet access)

```bash
pip install -r requirements.txt
```

Edit `config.py` to set your ticker (default: AAPL) and date range, then:

```bash
python train.py           # trains the LSTM, prints accuracy vs. naive baseline
python backtest.py        # runs training + simulates P&L, prints Sharpe/drawdown
python walk_forward.py    # the real test -- multiple sequential train/test folds
```

### Walk-forward validation (`walk_forward.py`)

A single train/test split can look great or terrible purely by luck of which
slice of history you happened to test on. `walk_forward.py` retrains the
model on an expanding window of history and tests on each subsequent block
of unseen data (5 folds by default), so you get 5 independent out-of-sample
readings instead of 1. Look at:

- **How many folds beat the accuracy baseline** -- a real edge should show up
  in most folds, not 1 out of 5.
- **Consistency across folds** -- wildly different results fold-to-fold
  usually means overfitting or that the model only works in certain market
  regimes, not a stable edge.

This is slower than a single `train.py` run since it trains N models instead
of one, but it's the step that tells you whether to trust the model at all.

### Multi-ticker + longer-horizon experiment (`multi_asset_experiment.py`)

If single-ticker, 1-day-direction walk-forward results don't show a
consistent edge, the next reasonable thing to try -- before concluding
there's no signal at all -- is giving the model more data and a less noisy
target:

```bash
python multi_asset_experiment.py
```

This trains ONE shared LSTM across all the tickers listed in
`config.TICKERS`, predicting `config.PREDICTION_HORIZON_DAYS`-day forward
direction (5 days by default) instead of next-day direction. Each ticker's
held-out test period is evaluated separately so you can see whether the
shared model works consistently across stocks or only on a lucky few --
same idea as walk-forward, but split by ticker instead of by time period.

**Decision rule**: if the model beats baseline on a clear majority of
tickers here, that's meaningfully more promising than the single-ticker
1-day result and worth investigating further (try even more tickers, tune
the horizon). If it's still only beating baseline on ~1 in N tickers, that's
strong evidence the direction-prediction approach itself doesn't have a real
edge with these features -- time to consider pivoting the project goal
(e.g. toward volatility prediction, which tends to be more learnable).

### Paper trading execution (`alpaca_execution.py`)

Once (and only once) walk-forward results look genuinely consistent, you can
connect the trained model to Alpaca's free paper trading API to test it on
live data with fake money:

1. Create a free account at [alpaca.markets](https://alpaca.markets) and
   generate **paper** trading API keys.
2. Set environment variables (never hardcode keys in the code):
   ```bash
   export ALPACA_API_KEY="your_paper_key"
   export ALPACA_SECRET_KEY="your_paper_secret"
   ```
3. Make sure `python train.py` has been run so a model exists.
4. Run `python alpaca_execution.py` -- it checks if the market is open,
   pulls recent bars, generates a signal, and places a paper order if the
   signal clears the confidence threshold, all within hard risk limits set
   in `config.py` (`MAX_POSITION_PCT`, `STOP_LOSS_PCT`).
5. To run it automatically, schedule it with cron (see the docstring at the
   top of `alpaca_execution.py` for an example crontab line). This script
   makes one decision per run and is meant to run once a day, not
   continuously.

**Risk limits are enforced in code, independent of the model** -- a stop-loss
check runs before the model's signal is even considered, and position size
is capped as a percentage of account equity. Do not remove these to "let the
model decide everything."

Real data will download automatically via `data_loader.py` -- you do NOT need
`generate_sample_data.py` on a machine with normal internet access. That script
only exists because this sandbox couldn't reach Yahoo Finance directly.

### Volatility prediction pivot (`train_volatility.py`, `backtest_volatility.py`)

After single-ticker (1/5 folds) and multi-ticker (2/8 tickers) walk-forward
results showed no consistent directional edge, this project pivoted to
predicting **volatility expansion** instead of price direction -- i.e. "will
this stock move more over the next N days than it has recently?" rather than
"will it go up or down?" This exploits volatility clustering, a much more
empirically robust pattern in markets than short-term direction.

```bash
python train_volatility.py       # trains the LSTM on the volatility-expansion label
python backtest_volatility.py    # translates it into a position-sizing action and evaluates
```

`backtest_volatility.py` does NOT try to beat buy-and-hold on raw return --
scaling down position size on predicted high-volatility days inherently
gives up some upside. Instead it checks two things:
1. Did reducing position size on flagged days improve max drawdown/risk by
   MORE than the return given up?
2. Did days flagged "high volatility predicted" actually have bigger real
   price moves than days flagged "low volatility"? (printed directly, not
   just accuracy -- this is the real-world test of whether the signal is
   worth acting on)

If both hold up consistently across tickers, this is a genuinely useful
risk-management signal -- even one that never says which direction to trade.

## How to interpret the results (important)

1. **Model accuracy alone means little.** Always compare against the "always
   predict the majority class" baseline printed by `train.py`. If your model
   isn't clearly beating that, it hasn't learned a real edge.
2. **Backtest return alone means little either.** Compare against buy-and-hold,
   printed by `backtest.py`. Beating a falling market by losing less isn't the
   same as making money.
3. **Test across multiple tickers and time periods** before trusting any result.
   A strategy that works on one stock/date-range is very likely overfit or lucky.
4. **Watch out for lookahead bias.** This pipeline splits train/test
   chronologically and fits the scaler only on training data specifically to
   avoid leaking future information -- don't undo that when you modify it.

## Suggested next steps, roughly in order

1. **Validate on real data**: run this on your machine with real AAPL (or another
   ticker) data and see how the model actually performs. ✅ pipeline ready
2. **Walk-forward validation**: `walk_forward.py` is built -- run it and look
   at consistency across folds, not just one number. ✅ built
3. **Paper trading**: `alpaca_execution.py` is built -- connect it once
   walk-forward results look consistent. ✅ built
4. **Add more tickers**: train a model that generalizes across many stocks
   rather than overfitting to one.
5. **Try a Transformer** instead of LSTM (e.g. a small time-series Transformer
   block) once the LSTM baseline is solid -- more data-hungry but can capture
   longer-range patterns.
6. **Automate retraining**: set up a periodic job (weekly/monthly) that
   reruns `train.py` on the latest data, since a static model trained once
   will drift out of date as market conditions change.
7. **Logging & alerting**: `alpaca_execution.py` currently just prints --
   consider writing structured logs to a file/database and alerting yourself
   (email/Slack) on trades or errors before running it unattended for long.

## Disclaimer

This project is for educational purposes. It is not financial advice, and
nothing here should be used to trade real money without extensive further
validation, risk management, and your own independent judgment. Trading
involves real risk of loss.