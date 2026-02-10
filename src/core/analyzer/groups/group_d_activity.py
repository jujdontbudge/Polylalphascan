"""Group D: Activity & Frequency Metrics (IDs 33-40)"""
from typing import Dict
import pandas as pd
import numpy as np

from .base_group import BaseMetricGroup, WalletData


class GroupDActivity(BaseMetricGroup):
    """Calculate activity and frequency metrics"""

    GROUP_ID = "D"
    METRIC_IDS = list(range(33, 41))  # 33-40

    def calculate(self, data: WalletData) -> Dict[int, float]:
        """Calculate all activity metrics"""
        results = {}
        df = data.trades_df

        if df.empty:
            return {mid: 0.0 for mid in self.METRIC_IDS}

        # Metric 33: Total Trade Count
        results[33] = len(df)

        # Metric 34: Total Volume (USD)
        results[34] = df['value'].sum()

        # Metric 35: Average Trade Size (USD)
        results[35] = df['value'].mean()

        # Calculate activity periods
        df['date'] = df['timestamp'].dt.date

        # Metric 36: Trades per Day
        first_date = df['timestamp'].min()
        last_date = df['timestamp'].max()
        days_span = (last_date - first_date).days + 1
        results[36] = self.safe_divide(len(df), days_span)

        # Metric 37: Active Trading Days
        active_days = df['date'].nunique()
        results[37] = active_days

        # Metric 38: Activity Ratio (%)
        results[38] = self.safe_divide(active_days, days_span) * 100

        # Metric 39: Unique Markets Traded
        unique_markets = df['market_id'].nunique()
        results[39] = unique_markets

        # Metric 40: Market Concentration (%)
        market_volumes = df.groupby('market_id')['value'].sum()
        max_market_volume = market_volumes.max() if not market_volumes.empty else 0
        total_volume = results[34]
        results[40] = self.safe_divide(max_market_volume, total_volume) * 100

        return results
