"""Group K: Copy-ability Metrics (IDs 95-100)"""
from typing import Dict
import pandas as pd
import numpy as np

from .base_group import BaseMetricGroup, WalletData


class GroupKCopyability(BaseMetricGroup):
    """Calculate copy-ability metrics for copy trading"""

    GROUP_ID = "K"
    METRIC_IDS = list(range(95, 101))  # 95-100

    def calculate(self, data: WalletData) -> Dict[int, float]:
        """Calculate all copy-ability metrics"""
        results = {}
        df = data.trades_df

        if df.empty:
            return {mid: 0.0 for mid in self.METRIC_IDS}

        # Metric 95: Trade Frequency Score
        # Good frequency for copying: not too fast, not too slow
        results[95] = self._calculate_frequency_score(df)

        # Metric 96: Execution Window (seconds)
        # Time available to copy trades
        results[96] = self._calculate_execution_window(df)

        # Metric 97: Liquidity Requirement
        # Min liquidity needed to copy
        results[97] = self._calculate_liquidity_requirement(df)

        # Metric 98: Copy Scalability Score
        # How well the strategy scales
        results[98] = self._calculate_scalability(df)

        # Metric 99: Transparency Score
        # Clarity of trading patterns
        results[99] = self._calculate_transparency(df)

        # Metric 100: Risk-Adjusted Copyability
        # Combined copy suitability metric
        pnl_series = self.calculate_pnl_series(df)
        returns = pnl_series / df['value']
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

        sharpe = self.safe_divide(
            returns.mean(),
            returns.std() if len(returns) > 1 else 0.01
        )

        results[100] = (
            results[99] * 0.3 +  # Transparency
            results[98] * 0.3 +  # Scalability
            min(100, (sharpe + 1) * 25) * 0.4  # Sharpe normalized
        )

        return results

    def _calculate_frequency_score(self, df: pd.DataFrame) -> float:
        """
        Calculate trade frequency suitability for copying.

        Ideal: 1-10 trades per day
        """
        if df.empty:
            return 0.0

        df_sorted = df.sort_values('timestamp')
        days_span = (df_sorted['timestamp'].max() - df_sorted['timestamp'].min()).days + 1
        trades_per_day = len(df) / days_span

        # Score based on trades/day
        if trades_per_day < 0.5:
            return 30  # Too slow
        elif trades_per_day <= 1:
            return 60
        elif trades_per_day <= 3:
            return 90  # Ideal
        elif trades_per_day <= 5:
            return 80
        elif trades_per_day <= 10:
            return 60
        elif trades_per_day <= 20:
            return 40
        else:
            return 20  # Too fast to copy

    def _calculate_execution_window(self, df: pd.DataFrame) -> float:
        """
        Calculate typical execution window in seconds.

        How long prices stay favorable after signal
        """
        if len(df) < 2:
            return 300.0  # Default 5 minutes

        # Estimate from time between trades in same market
        windows = []

        for market_id in df['market_id'].unique():
            market_df = df[df['market_id'] == market_id].sort_values('timestamp')

            if len(market_df) < 2:
                continue

            time_diffs = market_df['timestamp'].diff().dropna().dt.total_seconds()
            windows.extend(time_diffs.tolist())

        if not windows:
            return 300.0

        # Use median as typical window
        return np.median(windows)

    def _calculate_liquidity_requirement(self, df: pd.DataFrame) -> float:
        """
        Calculate minimum liquidity needed to copy.

        Based on average trade size * 2
        """
        if df.empty:
            return 0.0

        avg_size = df['value'].mean()
        return avg_size * 2

    def _calculate_scalability(self, df: pd.DataFrame) -> float:
        """
        Calculate how well the strategy scales.

        Higher variance in larger trades = less scalable
        """
        if len(df) < 10:
            return 50.0

        # Split into size quartiles
        df_sorted = df.sort_values('value')
        quartile_size = len(df) // 4

        if quartile_size < 2:
            return 50.0

        small_trades = df_sorted.head(quartile_size)
        large_trades = df_sorted.tail(quartile_size)

        # Calculate PnL ratio for small vs large
        small_pnl = self.calculate_pnl_series(small_trades)
        large_pnl = self.calculate_pnl_series(large_trades)

        small_win_rate = (small_pnl > 0).mean() if not small_pnl.empty else 0.5
        large_win_rate = (large_pnl > 0).mean() if not large_pnl.empty else 0.5

        # Similar performance across sizes = scalable
        win_rate_diff = abs(large_win_rate - small_win_rate)

        return max(0, 100 - win_rate_diff * 200)

    def _calculate_transparency(self, df: pd.DataFrame) -> float:
        """
        Calculate transparency/predictability of trading pattern.

        Higher = more predictable strategy
        """
        if len(df) < 5:
            return 50.0

        scores = []

        # 1. Consistent trade sizes
        size_cv = self.safe_divide(df['value'].std(), df['value'].mean())
        size_score = max(0, 100 - size_cv * 50)
        scores.append(size_score)

        # 2. Regular trading times
        df_sorted = df.sort_values('timestamp')
        time_diffs = df_sorted['timestamp'].diff().dropna().dt.total_seconds()
        if not time_diffs.empty and time_diffs.mean() > 0:
            time_cv = self.safe_divide(time_diffs.std(), time_diffs.mean())
            time_score = max(0, 100 - time_cv * 30)
            scores.append(time_score)

        # 3. Clear market focus
        market_concentration = df.groupby('market_id')['value'].sum()
        top_market_pct = market_concentration.max() / market_concentration.sum()
        # Too concentrated (>90%) or too spread (<10%) is bad
        if top_market_pct > 0.3 and top_market_pct < 0.9:
            focus_score = 80
        elif top_market_pct > 0.2 and top_market_pct < 0.95:
            focus_score = 60
        else:
            focus_score = 40
        scores.append(focus_score)

        # 4. Consistent direction
        buy_ratio = (df['side'] == 'BUY').mean()
        direction_clarity = abs(buy_ratio - 0.5) * 2  # 0 = 50/50, 1 = all one side
        direction_score = direction_clarity * 100
        scores.append(direction_score)

        return np.mean(scores)
