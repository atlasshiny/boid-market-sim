# Boid Market Simulator

A market simulation framework based on the **boids algorithm** (flocking behavior) that generates synthetic OHLCV data calibrated to real market statistics. This simulator is designed for generating directional synthetic augmentations of market data while preserving statistical properties observed in real market behavior.

## Overview

The Boid Market Simulator models market participants as autonomous agents ("boids") with distinct behavioral strategies:

- **Trend Agents** (alignment): Follow market momentum and herd behavior
- **Mean Reversion Agents** (separation): Push price back toward fair value
- **Noise Agents** (random): Introduce stochasticity and realistic market noise

The simulator uses three primary forces to drive price dynamics:
- **Alignment Force**: Trend agents align their momentum with neighbors, creating herding behavior
- **Separation Force**: Mean reversion agents push price back toward VWAP
- **Cohesion Force**: Fundamental agents pull price toward fair value

## Features

- **Real Data Calibration**: Compares synthetic output to real SPY data using statistical metrics (Hurst exponent, autocorrelation, volatility proxies)
- **Directional Augmentation**: Generate synthetic OHLCV data biased toward specific market regimes (directional or mean-reverting)
- **Flexible Agent Configuration**: Customize the mix of trend, mean-reversion, and noise agents
- **Format Support**: Load and save data in CSV or Parquet formats
- **CLI-based Workflow**: Easy-to-use command-line interface for calibration and generation

## Project Structure

```
src/boid-market-sim/
├── main.py           # CLI entrypoint
├── simulator.py      # Core simulator and agent classes
├── calibrate.py      # Calibration and augmentation utilities
├── run.bat          # Windows batch script to run the simulator
└── README.md        # This file
```

## Components

### `simulator.py`

**MarketBoid**: Represents a market participant with:
- `agent_type`: Type of agent ('trend', 'mean_rev', or 'noise')
- `position`: Current price belief
- `momentum`: Directional conviction [-1.0, 1.0]

**BoidsMarketSimulator**: Core simulator that:
- Manages a population of market boids
- Updates agent momentum based on alignment, separation, and cohesion forces
- Simulates price movements driven by aggregate agent behavior
- Generates synthetic OHLCV bars from close prices
- Parameters:
  - `n_trend` (300): Number of trend agents
  - `n_mean_rev` (300): Number of mean reversion agents
  - `n_noise` (400): Number of noise agents
  - `hurst_target` (0.6): Target Hurst exponent for calibration
  - `autocorr_target` (0.15): Target lag-1 autocorrelation
  - `seed`: Random seed for reproducibility

### `calibrate.py`

Utilities for working with real and simulated data:
- **`load_real_spy_csv()`**: Load OHLCV data from CSV or Parquet
- **`calibrate_simulator()`**: Compute calibration errors between real and simulated data
  - Compares: Hurst exponent, lag-1 autocorrelation, Garman-Klass volatility, Stochastic volatility
  - Returns relative absolute errors for each metric
- **`generate_directional_augmentation()`**: Generate synthetic OHLCV data biased toward trend or mean reversion

### `main.py`

CLI entrypoint that orchestrates the calibration and augmentation workflow.

## Usage

### Basic Calibration and Augmentation

```bash
python main.py \
  --real-data path/to/spy_data.csv \
  --n-bars 50000 \
  --n-synthetic 50000 \
  --output-dir ./output \
  --save-sim-csv \
  --save-calibration-json
```

### Configuration Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--real-data` | str | **Required** | Path to real SPY OHLCV CSV or Parquet file |
| `--n-bars` | int | 50000 | Number of bars used during calibration |
| `--n-synthetic` | int | 50000 | Number of synthetic bars to generate |
| `--trend-bias` | float | 0.6 | Bias toward trend agents (0.0-1.0); higher = more directional |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--n-trend` | int | 300 | Number of trend agents |
| `--n-mean-rev` | int | 300 | Number of mean reversion agents |
| `--n-noise` | int | 400 | Number of noise agents |
| `--initial-price` | float | None | Override initial price for synthetic generation (defaults to last real close) |
| `--output-dir` | str | None | Directory to save outputs |
| `--save-sim-csv` | flag | False | Save synthetic data as CSV |
| `--save-sim-parquet` | flag | False | Save synthetic data as Parquet |
| `--save-calibration-json` | flag | False | Save calibration errors as JSON |
| `--augmentation-only` | flag | False | Skip calibration; only generate synthetic augmentation |

### Example: Generate Trend-Biased Data

```bash
python main.py \
  --real-data data/spy_daily.csv \
  --trend-bias 0.8 \
  --n-synthetic 100000 \
  --output-dir ./augmented_data \
  --save-sim-csv
```

### Example: Calibration Only

```bash
python main.py \
  --real-data data/spy_daily.csv \
  --n-bars 50000 \
  --output-dir ./calibration \
  --save-calibration-json \
  --augmentation-only false  # This is the default
```

## Input Data Format

The simulator expects OHLCV data with the following columns:
- `open`: Opening price
- `high`: High price
- `low`: Low price
- `close`: Closing price
- `volume`: Trading volume

The index should be timestamps (automatically coerced to DatetimeIndex if needed).

## Output

The simulator generates:
1. **Synthetic OHLCV DataFrame** with:
   - `open`, `high`, `low`, `close`, `volume` columns
   - Same time alignment as input
   - Synthetic high/low derived from close ± noise

2. **Calibration JSON** (optional) with:
   - Relative absolute errors for calibration metrics
   - Enables verification that synthetic data matches real statistics

## Algorithm Details

### Price Update Mechanism

The price at each time step is updated based on:

```
dp = total_momentum * 0.005 + noise_force + news_shock
```

Where:
- `total_momentum`: Aggregate momentum across all agents
- `noise_force`: Random Gaussian noise (σ=0.002)
- `news_shock`: Exponential shocks (λ=0.001) occurring with 2% probability per bar

### Agent Momentum Updates

- **Trend Agents**: `momentum = 0.85 * momentum + 0.15 * alignment_force`
- **Mean Reversion Agents**: `momentum = 0.7 * momentum + 0.3 * separation_force`
- **Noise Agents**: `momentum = uniform(-1.0, 1.0)` (fully random)

## Calibration Metrics

The simulator is calibrated to match four statistical properties of real SPY data:

1. **Hurst Exponent** (target: 0.6): Measures persistence/mean reversion in price series
2. **Lag-1 Autocorrelation** (target: 0.15): Measures short-term momentum
3. **Garman-Klass Volatility** (gk_vol): Efficient volatility estimate
4. **Stochastic Volatility** (vol_stoch_k_12_3): Volatility clustering measure

## Use Cases

- **Data Augmentation**: Generate synthetic market data to increase training dataset size for ML models
- **Regime Simulation**: Create market scenarios with specific directional or mean-reverting characteristics
- **Strategy Backtesting**: Test trading algorithms on synthetic data with known statistical properties
- **Risk Analysis**: Generate market scenarios for stress testing

## Dependencies

The simulator requires:
- `numpy`: Numerical operations
- `pandas`: Data manipulation and OHLCV handling
- Optional: `pyarrow` for Parquet support

## Notes

- The simulator uses a default time step of `dt=1.0` (daily bars or other uniform intervals)
- Random state is seeded for reproducibility across runs
- The synthetic data is designed to match statistical properties of real SPY data but should not be used for actual trading
- Calibration is computationally lightweight and should complete in seconds for 50K bars
