import numpy as np
import pandas as pd

class MarketBoid:
    def __init__(self, agent_type: str, position: float, momentum: float):
        self.agent_type = agent_type  # 'trend', 'mean_rev', 'noise'
        self.position = position       # price belief
        self.momentum = momentum       # directional conviction [-1, 1]

class BoidsMarketSimulator:
    def __init__(
        self,
        n_trend: int = 300,       # alignment agents → Directional regime
        n_mean_rev: int = 300,    # separation agents → Side regime
        n_noise: int = 400,       # random agents → label noise
        dt: float = 1.0,
        hurst_target: float = 0.6,  # calibrate to match real SPY hurst_raw [2]
        autocorr_target: float = 0.15,  # calibrate to match real autocorr_1 [2]
        seed: int = 42,
    ):
        self.rng = np.random.default_rng(seed)
        self.n_trend = n_trend
        self.n_mean_rev = n_mean_rev
        self.n_noise = n_noise
        self.dt = dt
        self.hurst_target = hurst_target
        self.autocorr_target = autocorr_target
        self._init_agents()

    def _init_agents(self):
        self.agents = []
        for i in range(self.n_trend):
            self.agents.append(MarketBoid('trend', 0.0, self.rng.uniform(0.3, 0.8)))
        for i in range(self.n_mean_rev):
            self.agents.append(MarketBoid('mean_rev', 0.0, self.rng.uniform(-0.3, 0.3)))
        for i in range(self.n_noise):
            self.agents.append(MarketBoid('noise', 0.0, self.rng.uniform(-1.0, 1.0)))

    def _alignment(self, agents, radius: float = 0.05) -> float:
        """Trend agents align momentum with neighbors → herding."""
        momenta = [a.momentum for a in agents if a.agent_type == 'trend']
        return float(np.mean(momenta)) if momenta else 0.0

    def _separation(self, price: float, vwap: float) -> float:
        """Mean-rev agents push price back toward VWAP → reversion."""
        return -0.3 * (price - vwap)

    def _cohesion(self, price: float, fair_value: float) -> float:
        """Fundamental agents pull price toward fair value."""
        return -0.1 * (price - fair_value)

    def step(self, price: float, vwap: float, fair_value: float, news_shock: float = 0.0) -> float:
        alignment_force = self._alignment(self.agents)
        separation_force = self._separation(price, vwap)
        cohesion_force = self._cohesion(price, fair_value)
        noise_force = self.rng.normal(0, 0.002)

        # Update agent momenta
        for agent in self.agents:
            if agent.agent_type == 'trend':
                agent.momentum = 0.85 * agent.momentum + 0.15 * alignment_force
            elif agent.agent_type == 'mean_rev':
                agent.momentum = 0.7 * agent.momentum + 0.3 * separation_force
            else:
                agent.momentum = self.rng.uniform(-1.0, 1.0)

        # Aggregate price move
        total_momentum = np.mean([a.momentum for a in self.agents])
        dp = total_momentum * 0.005 + noise_force + news_shock
        return price + dp

    def simulate(self, n_bars: int, initial_price: float = 450.0) -> pd.DataFrame:
        prices = [initial_price]
        vwap = initial_price
        for i in range(n_bars - 1):
            # Hawkes process for news shocks
            news_shock = self.rng.exponential(0.001) if self.rng.random() < 0.02 else 0.0
            vwap = 0.99 * vwap + 0.01 * prices[-1]  # rolling vwap approx
            new_price = self.step(prices[-1], vwap, initial_price, news_shock)
            prices.append(new_price)

        close = pd.Series(prices)
        # Synthesize OHLCV from close
        noise = self.rng.normal(0, 0.002, n_bars)
        df = pd.DataFrame({
            'open':   close.shift(1).fillna(close.iloc[0]),
            'high':   close + np.abs(noise) * close,
            'low':    close - np.abs(noise) * close,
            'close':  close,
            'volume': self.rng.integers(1_000_000, 5_000_000, n_bars).astype(float),
        })
        return df