# Findings: AI Trading Bot Research (AAPL, 2015-2026)

This document summarizes what was actually tested, what the results were, and
what conclusions are and aren't supported by the evidence. Written to be
useful to a future you (or anyone else) picking this project back up.

## TL;DR

No tested approach produced a real, walk-forward-validated trading edge on
AAPL (or, for direction, across 8 large-cap tickers). Every promising-looking
single-split result was contradicted once tested properly across multiple
time periods or against a fair baseline. The one thing that held up
consistently was a weak, non-actionable statistical signal in the volatility
model's raw output -- real, but not strong enough to build a position-sizing
rule on.

## What was tried, in order

### 1. Direction prediction, single ticker (AAPL), 1-day horizon
- LSTM classifier, 14 technical indicators, 30-day lookback window
- Single train/test split: **model accuracy 49.3%, below the 54.9% majority
  baseline** -- and badly overfit (test loss rose while train loss fell)
- After adding early stopping + L2 regularization: accuracy improved to
  50.7%, still below baseline, but no longer a fake-confident overfit model
- **Walk-forward validation (5 folds): beat baseline in 1/5 folds.** Average
  strategy return (37.2%) underperformed buy-and-hold (49.4%).
- **Conclusion: no directional edge.**

### 2. Direction prediction, multi-ticker (8 large caps), 5-day horizon
- Hypothesis: more data (pooled across tickers) + less noisy target (5-day
  instead of 1-day) might surface a real edge
- One shared model trained across AAPL, MSFT, GOOGL, AMZN, META, NVDA, JPM, V
  (18,287 pooled training sequences)
- **Beat baseline in 2/8 tickers.** "Beats buy-and-hold" looked better (5/8)
  but the test period (mid-2024 to mid/late-2026) was a broad bull market
  across every ticker -- that number mostly reflects market beta, not skill.
- **Conclusion: no directional edge here either.** More data and a longer
  horizon didn't change the fundamental finding.

### 3. Volatility expansion prediction (pivot from direction)
- Hypothesis: volatility clustering is a more robust, better-documented
  pattern than short-term price direction (this is the basis of GARCH models)
- Label: will realized volatility over the next 5 days exceed the trailing
  10-day volatility?
- Single split: **58.7% accuracy vs. 55.6% baseline** -- the best single-split
  margin seen anywhere in this project. Precision on the "Expand" class was
  57% vs. a 44% base rate.
- **Walk-forward (5 folds): beat baseline in 2/5 folds.** Weaker than the
  single split suggested, per the pattern established earlier in the project.

### 4. Volatility -> position-sizing action, binary threshold
- Translated the volatility signal into a concrete action: scale position to
  50% on predicted high-vol days, full position otherwise
- First pass (vs. always-100%-invested baseline): **drawdown improved in
  5/5 folds**, looked very promising
- **Critical fix: added a fair "constant exposure" control** -- same average
  exposure level, applied uniformly instead of selectively. This controls for
  the fact that being under-invested on average mechanically reduces both
  drawdown and return, regardless of which specific days you pick.
- **Against the fair control: beat it in only 1/5 folds.** Most of the
  apparent benefit was just an artifact of lower average exposure (which
  ranged 64.5%-96.6% across folds), not real day-by-day timing skill.

### 5. Volatility -> position-sizing action, graded (continuous)
- Hypothesis: a binary 0.5 threshold discards the graded, ordinal
  information in the model's raw probability output -- maybe using
  probability continuously (position = 1 - 0.5 x prob) captures value the
  threshold version missed
- **Result: also beat the fair control in only 1/5 folds.** Same result as
  the binary version. This idea didn't pan out.

## The one thing that held up

Across every single fold, in both the binary and graded volatility
experiments (10/10 total fold-checks), **days the model flagged as
"predicted expand" showed a more positive actual volatility change than days
flagged "predicted contract/flat."** This is a real, consistent, correctly-
signed statistical signal in the model's raw output. It just isn't strong
enough, once turned into any position-sizing rule tested here, to beat the
trivial strategy of holding a constant smaller position all the time.

## What this project demonstrates methodologically

Several promising-looking results appeared and were each overturned by a
more rigorous check, in order:
1. A single train/test split looking fine, corrected by walk-forward
   validation (revealed 1/5 fold performance, not consistent skill)
2. A backtest showing "drawdown improved," corrected by comparing against a
   fair constant-exposure baseline (revealed the improvement was mostly
   mechanical, not timing skill)
3. A confidence threshold silently preventing trades in some folds, fixed
   by exposing the raw probability distribution per fold

This pattern -- promising result, then a sharper test reveals it doesn't
hold up -- happened multiple times and is the expected, healthy outcome of
doing this kind of validation properly. It is much better to find this out
in a rigorous backtest than after risking real (or even paper) capital on a
model that looked good on one lucky split.

## Honest options for future work

- **Try a genuinely different feature set**: order book/microstructure data,
  options-implied volatility (VIX-style), sentiment/news data, or
  macroeconomic indicators -- the 14 technical indicators used throughout
  this project may simply not contain enough information, regardless of
  architecture.
- **Try the "one real finding" more seriously**: the consistent
  correctly-signed volatility signal is real but weak. A more sophisticated
  translation into an action (e.g. options strategies that specifically
  profit from correctly-ranked but not perfectly-calibrated volatility views,
  rather than simple position sizing) might extract more value from it than
  anything tried here.
- **Test on a broader universe / longer history**: everything here used
  daily bars on ~10 years of data for at most 8 tickers. A much larger
  cross-section (hundreds of tickers) is more where deep learning approaches
  in quant finance typically start to show real value, if it exists at all
  for retail-accessible data.
- **Treat this as a genuinely negative result and move on**: this is also a
  completely legitimate conclusion. Not every reasonable hypothesis pans
  out, and knowing that with this much rigor is valuable information in
  itself.

## Project artifacts (for reference)

All code, config, and this document live in this project folder. Nothing
here is connected to real money; `alpaca_execution.py` targets Alpaca's
PAPER trading API only, and even that has not been validated against any of
the models built here, since none of them cleared the bar for real use.