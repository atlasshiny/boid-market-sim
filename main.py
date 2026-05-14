from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

"""CLI entrypoint for the Boids market simulator calibration and augmentation.

This script exposes two small workflows used during experiment development:
- Calibrate the simulator to a real SPY OHLCV frame (CSV or Parquet).
- Generate directional synthetic OHLCV augmentations using the calibrated
    simulator and save them as CSV or Parquet.

The module keeps imports local-friendly so it can be executed from the
repository root or from the package directory.
"""

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
for candidate in (str(HERE), str(REPO_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from calibrate import calibrate_simulator, generate_directional_augmentation, load_real_spy_csv
from simulator import BoidsMarketSimulator

def parse_args() -> argparse.Namespace:
    """Build and return the CLI argument parser.

    The CLI accepts a path to a real OHLCV dataset (CSV or Parquet), tuning
    parameters for the simulator, and flags to persist the generated outputs.
    """
    parser = argparse.ArgumentParser(description="Calibrate the boid-based market simulator and optionally generate synthetic data.")
    # Path to a real SPY OHLCV file. Support CSV and Parquet inputs.
    parser.add_argument("--real-data", type=str, required=True, help="Path to a real SPY OHLCV CSV or Parquet file")
    parser.add_argument("--n-bars", type=int, default=50_000, help="Number of bars used during calibration")
    parser.add_argument("--n-synthetic", type=int, default=50_000, help="Number of synthetic bars to generate")
    parser.add_argument("--trend-bias", type=float, default=0.6, help="Bias toward directional/trend agents for augmentation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the simulator")
    parser.add_argument("--n-trend", type=int, default=300, help="Number of trend agents")
    parser.add_argument("--n-mean-rev", type=int, default=300, help="Number of mean-reversion agents")
    parser.add_argument("--n-noise", type=int, default=400, help="Number of noise agents")
    parser.add_argument("--initial-price", type=float, default=None, help="Override initial price for synthetic generation")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional directory to save calibration JSON and synthetic CSV")
    parser.add_argument("--save-sim-csv", action="store_true", help="Save the synthetic sample as CSV")
    parser.add_argument("--save-sim-parquet", action="store_true", help="Save the synthetic sample as Parquet")
    parser.add_argument("--save-calibration-json", action="store_true", help="Save calibration errors as JSON")
    parser.add_argument("--augmentation-only", action="store_true", help="Only generate the synthetic augmentation CSV")
    return parser.parse_args()

def main() -> None:
    # Parse CLI args and load the provided real SPY OHLCV frame.
    args = parse_args()
    # `load_real_spy_csv` transparently handles CSV or Parquet input and
    # returns a time-indexed DataFrame suitable for calibration.
    real_spy_df = load_real_spy_csv(args.real_data)

    sim = BoidsMarketSimulator(
        n_trend=args.n_trend,
        n_mean_rev=args.n_mean_rev,
        n_noise=args.n_noise,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    if not args.augmentation_only:
        # Run a short calibration that compares a handful of summary
        # statistics (hurst, autocorr, volatility proxies) between the
        # real data and a synthetic run of the simulator.
        calibration = calibrate_simulator(sim, real_spy_df, n_bars=args.n_bars)
        print("Calibration error:")
        print(json.dumps(calibration, indent=2))
        if output_dir is not None and args.save_calibration_json:
            calibration_path = output_dir / "calibration_error.json"
            calibration_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
            print(f"Saved calibration JSON to {calibration_path}")

    # Determine the starting price for synthetic generation. If the user
    # provided `--initial-price` we use that, otherwise we use the last
    # observed real close.
    initial_price = float(args.initial_price) if args.initial_price is not None else float(real_spy_df["close"].iloc[-1])

    # Generate the synthetic augmentation biased toward trend/mean-reversion
    # according to `--trend-bias`.
    synthetic_df = generate_directional_augmentation(
        real_spy_df,
        n_synthetic=args.n_synthetic,
        trend_bias=args.trend_bias,
        initial_price=initial_price,
    )

    print(f"Generated synthetic bars: {len(synthetic_df)} rows")
    print(synthetic_df.head().to_string())

    if output_dir is not None and args.save_sim_csv:
        sim_path = output_dir / "synthetic_bars.csv"
        synthetic_df.to_csv(sim_path, index=True)
        print(f"Saved synthetic CSV to {sim_path}")
    if output_dir is not None and args.save_sim_parquet:
        pq_path = output_dir / "synthetic_bars.parquet"
        # to_parquet may not preserve index name; keep index so timestamps are retained
        synthetic_df.to_parquet(pq_path, index=True)
        print(f"Saved synthetic Parquet to {pq_path}")

if __name__ == "__main__":
    main()