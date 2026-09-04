"""
Same walk-forward validation as walk_forward_volatility.py, but using
continuous (graded-by-probability) position sizing instead of a binary
0.5-threshold decision. See backtest_volatility_graded.py for the reasoning.

This is the real test: does graded sizing beat the constant-exposure control
in a clear majority of folds, where the binary threshold version only
managed 1/5?
"""

from walk_forward_volatility import run_volatility_walk_forward
from backtest_volatility import simulate_position_scaling_graded

if __name__ == "__main__":
    run_volatility_walk_forward(scaling_fn=simulate_position_scaling_graded)