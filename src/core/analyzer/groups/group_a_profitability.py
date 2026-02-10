"""Group A: Profitability Metrics (IDs 1-10)"""
from typing import Dict
import pandas as pd
import numpy as np

from .base_group import BaseMetricGroup, WalletData


class GroupAProfitability(BaseMetricGroup):
    """Calculate profitability metrics"""

    GROUP_ID = "A"
    METRIC_IDS = list(range(1, 11))  # 1-10

    def calculate(self, data: WalletData) -> Dict[int, float]:
        """Calculate all profitability metrics"""
        results = {}
        df = data.trades_df

        if df.empty:
            return {mid: 0.0 for mid in self.METRIC_IDS}

        # Use memoized market_pnl_map from WalletData (prevents duplicate calculation)
        market_pnls = data.market_pnl_map
        total_pnl = sum(market_pnls.values())

        # Metric 1: Total PnL (USD) - Uses corrected market-based calculation
        results[1] = total_pnl

        # Metric 2: Total ROI (%)
        # ROI = PnL / Total Buy Cost (not total volume which includes sells)
        buys = df[df['side'] == 'BUY']
        total_invested = buys['value'].sum() if not buys.empty else 0.0
        results[2] = self.safe_divide(total_pnl, total_invested) * 100

        # Metric 3: Annualized ROI (%)
        if not df.empty and len(df) > 1:
            days_active = (df['timestamp'].max() - df['timestamp'].min()).days
            if days_active > 0:
                roi = self.safe_divide(total_pnl, total_invested)
                annualized = ((1 + roi) ** (365 / days_active) - 1) * 100
                results[3] = min(max(annualized, -100), 1000)  # Cap extremes
            else:
                results[3] = results[2]
        else:
            results[3] = 0.0

        # Metric 4: Profit Factor (winning market PnLs / losing market PnLs)
        winning_pnl = sum(pnl for pnl in market_pnls.values() if pnl > 0)
        losing_pnl = abs(sum(pnl for pnl in market_pnls.values() if pnl < 0))
        results[4] = self.safe_divide(winning_pnl, losing_pnl, default=1.0)

        # Metric 5: Average Market PnL
        if market_pnls:
            results[5] = sum(market_pnls.values()) / len(market_pnls)
        else:
            results[5] = 0.0

        # Metric 6: Median Market PnL
        if market_pnls:
            pnl_values = sorted(market_pnls.values())
            mid = len(pnl_values) // 2
            if len(pnl_values) % 2 == 0:
                results[6] = (pnl_values[mid - 1] + pnl_values[mid]) / 2
            else:
                results[6] = pnl_values[mid]
        else:
            results[6] = 0.0

        # Metric 7: Best Market PnL
        results[7] = max(market_pnls.values()) if market_pnls else 0.0

        # Metric 8: Worst Market PnL
        results[8] = min(market_pnls.values()) if market_pnls else 0.0

        # Metric 9: PnL per Volume
        total_volume = df['value'].sum()
        results[9] = self.safe_divide(total_pnl, total_volume)

        # Metric 10: Realized vs Unrealized Ratio
        # Use absolute values only in denominator for magnitude, preserve sign in numerator
        realized_pnl = data.positions_df['realized_pnl'].sum() if not data.positions_df.empty else 0
        unrealized_pnl = data.positions_df['unrealized_pnl'].sum() if not data.positions_df.empty else 0
        total_magnitude = abs(realized_pnl) + abs(unrealized_pnl)
        results[10] = self.safe_divide(realized_pnl, total_magnitude, default=1.0)

        return results

