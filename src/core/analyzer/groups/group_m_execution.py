"""Group M: Execution Quality Metrics (IDs 105-108) [NEW IN v2.0]"""
from typing import Dict
import pandas as pd
import numpy as np

from .base_group import BaseMetricGroup, WalletData


class GroupMExecution(BaseMetricGroup):
    """Calculate execution quality metrics"""

    GROUP_ID = "M"
    METRIC_IDS = list(range(105, 109))  # 105-108

    def calculate(self, data: WalletData) -> Dict[int, float]:
        """Calculate all execution quality metrics"""
        results = {}
        df = data.trades_df

        if df.empty:
            return {mid: 0.0 for mid in self.METRIC_IDS}

        # Metric 105: Fill Rate (%)
        results[105] = self._calculate_fill_rate(df)

        # Metric 106: Price Improvement (bps)
        results[106] = self._calculate_price_improvement(df)

        # Metric 107: Execution Speed Score
        results[107] = self._calculate_execution_speed(df)

        # Metric 108: Market Impact (bps)
        results[108] = self._calculate_market_impact(df)

        return results

    def _calculate_fill_rate(self, df: pd.DataFrame) -> float:
        """
        Calculate fill rate.

        Without order data, estimate from trade patterns.
        Consistent fills suggest good fill rate.
        """
        if df.empty:
            return 100.0  # Assume all trades are filled

        # Proxy: check for fragmented fills (multiple small trades in succession)
        df_sorted = df.sort_values('timestamp')

        # Group by market and check for quick successive trades
        fragmented_count = 0
        total_count = len(df)

        for market_id in df['market_id'].unique():
            market_df = df_sorted[df_sorted['market_id'] == market_id]

            if len(market_df) < 2:
                continue

            time_diffs = market_df['timestamp'].diff().dt.total_seconds()

            # Trades < 10 seconds apart in same direction might be partial fills
            for i in range(1, len(market_df)):
                if time_diffs.iloc[i] < 10:
                    if market_df.iloc[i]['side'] == market_df.iloc[i-1]['side']:
                        fragmented_count += 1

        # Higher fragmentation = potentially lower original fill rate
        fragmentation_rate = self.safe_divide(fragmented_count, total_count)
        return max(70, 100 - fragmentation_rate * 50)

    def _calculate_price_improvement(self, df: pd.DataFrame) -> float:
        """
        Calculate price improvement in basis points.

        Better execution than mid-price.
        """
        if df.empty:
            return 0.0

        improvements = []

        for _, trade in df.iterrows():
            price = trade['price']
            side = trade['side']

            # Estimate mid-price as 0.5 for binary markets
            # In reality, would compare to actual order book mid
            mid_price = 0.5

            if side == 'BUY':
                # Improvement = mid - execution (lower is better for buys)
                improvement = (mid_price - price) / mid_price * 10000
            else:
                # Improvement = execution - mid (higher is better for sells)
                improvement = (price - mid_price) / mid_price * 10000

            improvements.append(improvement)

        return np.mean(improvements) if improvements else 0

    def _calculate_execution_speed(self, df: pd.DataFrame) -> float:
        """
        Calculate execution speed score.

        Based on consistency of trade timing.
        """
        if len(df) < 3:
            return 50.0

        df_sorted = df.sort_values('timestamp')

        # Calculate time from potential signal to execution
        # Proxy: time between trades (faster = better execution)
        time_diffs = df_sorted['timestamp'].diff().dropna().dt.total_seconds()

        if time_diffs.empty:
            return 50.0

        # Score based on median execution time
        median_time = time_diffs.median()

        # Convert to score: < 1 min = 100, > 1 hour = 20
        if median_time < 60:
            return 100
        elif median_time < 300:  # 5 minutes
            return 85
        elif median_time < 900:  # 15 minutes
            return 70
        elif median_time < 3600:  # 1 hour
            return 50
        else:
            return max(20, 50 - (median_time - 3600) / 3600 * 10)

    def _calculate_market_impact(self, df: pd.DataFrame) -> float:
        """
        Calculate market impact in basis points.

        Price movement caused by trades (lower is better).
        """
        if len(df) < 3:
            return 0.0

        impacts = []

        for market_id in df['market_id'].unique():
            market_df = df[df['market_id'] == market_id].sort_values('timestamp')

            if len(market_df) < 2:
                continue

            for i in range(len(market_df) - 1):
                pre_price = market_df.iloc[i]['price']
                post_price = market_df.iloc[i + 1]['price']
                trade_value = market_df.iloc[i]['value']

                # Impact = price change * trade size relationship
                price_change = abs(post_price - pre_price) / pre_price if pre_price > 0 else 0
                impact_bps = price_change * 10000

                # Weight by trade size (larger trades should have more impact)
                impacts.append(impact_bps)

        return np.mean(impacts) if impacts else 0
