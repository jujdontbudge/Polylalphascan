"""Group J: Hygiene & Metadata Metrics (IDs 84-94)"""
from typing import Dict
from datetime import datetime, timezone
import pandas as pd
import numpy as np

from .base_group import BaseMetricGroup, WalletData


class GroupJHygiene(BaseMetricGroup):
    """Calculate hygiene and metadata metrics"""

    GROUP_ID = "J"
    METRIC_IDS = list(range(84, 95))  # 84-94

    def calculate(self, data: WalletData) -> Dict[int, float]:
        """Calculate all hygiene metrics"""
        results = {}
        df = data.trades_df

        if df.empty:
            return {mid: 0.0 for mid in self.METRIC_IDS}

        now = datetime.now(timezone.utc).replace(tzinfo=None)

        # Metric 84: Wallet Age (days)
        first_trade = df['timestamp'].min()
        if pd.notna(first_trade):
            wallet_age = (now - first_trade.to_pydatetime()).days
            results[84] = wallet_age
        else:
            results[84] = 0

        # Metric 85: Days Since Last Trade
        last_trade = df['timestamp'].max()
        if pd.notna(last_trade):
            days_since = (now - last_trade.to_pydatetime()).days
            results[85] = days_since
        else:
            results[85] = 0

        # Metric 86: Trading History Length
        if pd.notna(first_trade) and pd.notna(last_trade):
            history_length = (last_trade - first_trade).days
            results[86] = history_length
        else:
            results[86] = 0

        # Metric 87: Data Completeness (%)
        # Check for missing values in key fields
        key_fields = ['timestamp', 'market_id', 'side', 'price', 'value']
        complete_rows = df[key_fields].notna().all(axis=1).sum()
        results[87] = self.safe_divide(complete_rows, len(df)) * 100

        # Metric 88: Verified Positions (%)
        # Check if positions reconcile with trade history
        results[88] = self._calculate_position_verification(data)

        # Metric 89: Trade Anomaly Rate (%)
        results[89] = self._calculate_anomaly_rate(df)

        # Metric 90: Failed Trade Rate (%)
        # This would need tx status; estimate from patterns
        results[90] = 0  # Placeholder - needs tx data

        # Metric 91: Gas Efficiency Score
        # For Polymarket, this is less relevant; use fee efficiency
        total_fees = df['fee'].sum()
        total_value = df['value'].sum()
        fee_ratio = self.safe_divide(total_fees, total_value)
        results[91] = max(0, 100 - fee_ratio * 1000)  # Lower fees = higher score

        # Metric 92: Average Slippage (%)
        results[92] = self._estimate_slippage(df)

        # Metric 93: Cross-Market Activity
        # Number of markets with correlated activity
        results[93] = self._calculate_cross_market_activity(df)

        # Metric 94: Data Quality Score
        # Composite of above metrics
        results[94] = (
            results[87] * 0.3 +
            results[88] * 0.3 +
            (100 - results[89]) * 0.2 +
            results[91] * 0.2
        )

        return results

    def _calculate_position_verification(self, data: WalletData) -> float:
        """
        Verify positions against trade history.

        Returns percentage of positions that reconcile.
        """
        if data.positions_df.empty or data.trades_df.empty:
            return 100.0  # No positions to verify

        verified = 0
        total = len(data.positions_df)

        for _, pos in data.positions_df.iterrows():
            market_id = pos['market_id']
            market_trades = data.trades_df[data.trades_df['market_id'] == market_id]

            if not market_trades.empty:
                # Simple check: has trades for this market
                verified += 1

        return self.safe_divide(verified, total) * 100 if total > 0 else 100

    def _calculate_anomaly_rate(self, df: pd.DataFrame) -> float:
        """
        Calculate rate of anomalous trades.

        Anomalies: unusual size, price, or timing
        """
        if len(df) < 10:
            return 0.0

        anomalies = 0

        # Size anomalies (> 3 std from mean)
        size_mean = df['value'].mean()
        size_std = df['value'].std()
        if size_std > 0:
            size_anomalies = ((df['value'] - size_mean).abs() > 3 * size_std).sum()
            anomalies += size_anomalies

        # Price anomalies (outside 0-1 range for binary)
        price_anomalies = ((df['price'] < 0) | (df['price'] > 1)).sum()
        anomalies += price_anomalies

        # Time anomalies (< 1 second between trades)
        df_sorted = df.sort_values('timestamp')
        time_diffs = df_sorted['timestamp'].diff().dt.total_seconds()
        time_anomalies = (time_diffs < 1).sum()
        anomalies += time_anomalies

        return self.safe_divide(anomalies, len(df) * 3) * 100  # Normalize by checks

    def _estimate_slippage(self, df: pd.DataFrame) -> float:
        """
        Estimate average slippage.

        Slippage = deviation from round prices
        """
        if df.empty:
            return 0.0

        # For each trade, check deviation from nearest round price (0.1 increments)
        deviations = []

        for _, trade in df.iterrows():
            price = trade['price']
            nearest_round = round(price * 10) / 10
            deviation = abs(price - nearest_round) / price if price > 0 else 0
            deviations.append(deviation)

        return np.mean(deviations) * 100 if deviations else 0

    def _calculate_cross_market_activity(self, df: pd.DataFrame) -> int:
        """
        Calculate cross-market activity.

        Count markets with similar timing patterns.
        """
        if len(df) < 5:
            return 0

        markets = df['market_id'].unique()
        if len(markets) < 2:
            return 0

        # Count pairs of markets with overlapping activity
        cross_pairs = 0

        for i, m1 in enumerate(markets):
            for m2 in markets[i+1:]:
                m1_df = df[df['market_id'] == m1]
                m2_df = df[df['market_id'] == m2]

                # Check for temporal overlap (trades within 1 hour of each other)
                for _, t1 in m1_df.iterrows():
                    for _, t2 in m2_df.iterrows():
                        time_diff = abs((t1['timestamp'] - t2['timestamp']).total_seconds())
                        if time_diff < 3600:  # 1 hour
                            cross_pairs += 1
                            break
                    else:
                        continue
                    break

        return cross_pairs
