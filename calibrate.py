from __future__ import annotations

"""Boid market simulator calibration helpers.

This module provides utilities used by the CLI to:
- load a real SPY OHLCV frame (CSV or Parquet),
- compute a small set of calibration statistics between the real
  data and simulated output, and
- generate directional synthetic augmentations.

The helpers are intentionally lightweight and accept pandas DataFrames
so they remain easy to test.
"""

from pathlib import Path
import sys

import pandas as pd

try:
    from simulator import BoidsMarketSimulator
except ModuleNotFoundError:
    # When running as a script inside the package directory the import
    # may fail; fall back to a local import path so the module is import-safe.
    _HERE = Path(__file__).resolve().parent
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    from simulator import BoidsMarketSimulator

def _load_feature_frame(df: pd.DataFrame, n_bars: int) -> pd.DataFrame:
    """Normalize and validate an OHLCV frame.

    - Ensures the index is a DatetimeIndex (coerces if needed).
    - Sorts by time and returns the last ``n_bars`` rows.
    - Verifies required OHLCV columns are present.
    """
    frame = df.copy()
    # Make sure index is a datetime index for downstream feature code.
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index, errors="coerce")

    # Keep only the most recent `n_bars` records.
    frame = frame.sort_index().tail(n_bars)

    # Validate presence of OHLCV columns expected by compute_default_math_features
    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"real_spy_df is missing required OHLCV columns: {missing}")

    return frame

def calibrate_simulator(sim: BoidsMarketSimulator, real_spy_df: pd.DataFrame, n_bars: int = 50000) -> dict:
    """Compute a small set of calibration errors between real and simulated data.

    The function computes derived math features for both the simulated and
    real data using `compute_default_math_features` and then returns a
    dictionary of relative absolute errors for a few summary statistics
    (hurst, lag-1 autocorrelation, and two volatility proxies).
    """
    from src.data_prep import compute_default_math_features

    # Generate synthetic bars and compute features for both series.
    sim_df = sim.simulate(n_bars)
    sim_features = compute_default_math_features(sim_df)
    real_features = compute_default_math_features(_load_feature_frame(real_spy_df, n_bars))

    # Compare a handful of moments used by the model and report relative errors.
    calibration_error = {}
    for col in ["hurst_raw", "autocorr_1", "gk_vol", "vol_stoch_k_12_3"]:
        real_mean = float(real_features[col].mean())
        sim_mean = float(sim_features[col].mean())
        # Relative absolute difference (small epsilon to avoid division by zero)
        calibration_error[col] = abs(real_mean - sim_mean) / (abs(real_mean) + 1e-8)

    return calibration_error

def generate_directional_augmentation(
    real_spy_df: pd.DataFrame,
    n_synthetic: int = 50000,
    trend_bias: float = 0.6,  # push toward Directional regime
    initial_price: float | None = None,
) -> pd.DataFrame:
    """Generate a synthetic OHLCV augmentation biased toward trend.

    The function constructs a simulator with a configurable balance of
    trend/mean-reversion agents (controlled by ``trend_bias``) and returns
    a DataFrame of synthetic OHLCV bars starting from ``initial_price``
    (defaults to the last close in ``real_spy_df``).
    """
    sim = BoidsMarketSimulator(
        n_trend=int(1000 * trend_bias),
        n_mean_rev=int(1000 * (1 - trend_bias)),
        n_noise=400,
    )

    # Use the provided initial_price or fall back to the last real close.
    start_price = float(initial_price if initial_price is not None else real_spy_df["close"].iloc[-1])
    sim_df = sim.simulate(n_synthetic, initial_price=start_price)
    return sim_df

def load_real_spy_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load a real SPY OHLCV CSV or Parquet and normalize it for calibration.

    Supports CSV and Parquet files. The returned DataFrame will be indexed
    by a DatetimeIndex derived from a `timestamp`/`date`/first column if present.
    """
    path = Path(csv_path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".parq", ".pq"}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.set_index("timestamp")
    elif df.columns.size > 0 and str(df.columns[0]).lower() in {"date", "datetime", "time"}:
        first_col = df.columns[0]
        df[first_col] = pd.to_datetime(df[first_col], errors="coerce")
        df = df.set_index(first_col)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    df = df.sort_index()
    return df