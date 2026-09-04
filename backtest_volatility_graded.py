"""
Same volatility model, backtest, and fair constant-exposure control as
backtest_volatility.py -- the only difference is the position-sizing rule:
continuous (graded by predicted probability) instead of a binary 0.5
threshold with a fixed scale-down.

This exists to test one specific idea: the binary-threshold version throws
away the graded, ordinal information in the model's raw probability output.
evaluate_volatility_prediction_quality showed that information holding up
consistently (5/5 folds) even when the binary threshold decision didn't --
this checks whether using it directly, without discretizing, captures real
value the threshold version missed.

Run this, then walk_forward_volatility_graded.py for the real (multi-fold) test.
"""

from backtest_volatility import run_volatility_backtest, simulate_position_scaling_graded

if __name__ == "__main__":
    run_volatility_backtest(scaling_fn=simulate_position_scaling_graded)