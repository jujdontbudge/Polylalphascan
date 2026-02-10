"""Group F: Money Management Metrics (IDs 49-57)"""
from typing import Dict
import pandas as pd
import numpy as np

from .base_group import BaseMetricGroup, WalletData


class GroupFMoney(BaseMetricGroup):
    """Calculate money management metrics"""

    GROUP_ID = "F"
    METRIC_IDS = list(range(49, 58))  # 49-57

    def calculate(self, data: WalletData) -> Dict[int, float]:
        """Calculate all money management metrics"""
        results = {}
        df = data.trades_df

        if df.empty:
            return {mid: 0.0 for mid in self.METRIC_IDS}

        # Estimate portfolio value over time
        total_volume = df['value'].sum()
        avg_trade_size = df['value'].mean()
        estimated_portfolio = total_volume / 10  # Rough estimate

        # Metric 49: Average Position Size (%)
        position_sizes = df['value'] / estimated_portfolio * 100
        results[49] = position_sizes.mean()

        # Metric 50: Max Position Size (%)
        results[50] = position_sizes.max()

        # Metric 51: Position Size Std Dev
        results[51] = position_sizes.std() if len(position_sizes) > 1 else 0

        # Metric 52: Avg Risk per Trade (%)
        # Estimated as position size * (1 - price) for buys
        df_copy = df.copy()
        df_copy['risk'] = df_copy.apply(
            lambda r: r['value'] * (1 - r['price']) if r['side'] == 'BUY' else r['value'] * r['price'],
            axis=1
        )
        avg_risk = df_copy['risk'].mean()
        results[52] = self.safe_divide(avg_risk, estimated_portfolio) * 100

        # Metric 53: Average Leverage
        # For Polymarket, no real leverage, but can estimate effective leverage
        results[53] = 1.0  # Default to 1x

        # Metric 54: Maximum Leverage
        results[54] = 1.0  # Default to 1x

        # Metric 55: Diversification Score
        # Based on market concentration (inverse)
        market_volumes = df.groupby('market_id')['value'].sum()
        weights = market_volumes / market_volumes.sum()
        herfindahl = (weights ** 2).sum()
        results[55] = (1 - herfindahl) * 100  # 0-100 scale

        # Metric 56: Concentration Risk (%)
        # Top 3 markets percentage
        top_3_volume = market_volumes.nlargest(3).sum()
        results[56] = self.safe_divide(top_3_volume, total_volume) * 100

        # Metric 57: Capital Efficiency
        pnl_series = self.calculate_pnl_series(df)
        total_pnl = pnl_series.sum()
        avg_capital = df['value'].expanding().mean().mean()
        results[57] = self.safe_divide(total_pnl, avg_capital)

        return results
