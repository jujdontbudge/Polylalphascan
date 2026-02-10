"""Group E: Timing & Speed Metrics (IDs 41-48)"""
from typing import Dict
import pandas as pd
import numpy as np

from .base_group import BaseMetricGroup, WalletData


class GroupETiming(BaseMetricGroup):
    """Calculate timing and speed metrics"""

    GROUP_ID = "E"
    METRIC_IDS = list(range(41, 49))  # 41-48

    def calculate(self, data: WalletData) -> Dict[int, float]:
        """Calculate all timing metrics"""
        results = {}
        df = data.trades_df

        if df.empty:
            return {mid: 0.0 for mid in self.METRIC_IDS}

        # Calculate hold times by matching BUY/SELL pairs
        hold_times = self._calculate_hold_times(df)
        pnl_series = self.calculate_pnl_series(df)

        # Metric 41: Average Hold Time (hours)
        results[41] = hold_times.mean() if not hold_times.empty else 0

        # Metric 42: Median Hold Time (hours)
        results[42] = hold_times.median() if not hold_times.empty else 0

        # Metric 43: Minimum Hold Time (seconds)
        results[43] = hold_times.min() * 3600 if not hold_times.empty else 0  # Convert hours to seconds

        # Metric 44: Maximum Hold Time (days)
        results[44] = hold_times.max() / 24 if not hold_times.empty else 0  # Convert hours to days

        # Calculate winning/losing hold times
        df_with_pnl = df.copy()
        df_with_pnl['pnl'] = pnl_series

        winning_holds, losing_holds = self._split_hold_times(df_with_pnl)

        # Metric 45: Avg Time to Profit (hours)
        results[45] = winning_holds.mean() if not winning_holds.empty else 0

        # Metric 46: Avg Time to Loss (hours)
        results[46] = losing_holds.mean() if not losing_holds.empty else 0

        # Metric 47: Entry Timing Score (0-100)
        results[47] = self._calculate_entry_timing_score(df)

        # Metric 48: Exit Timing Score (0-100)
        results[48] = self._calculate_exit_timing_score(df)

        return results

    def _calculate_hold_times(self, df: pd.DataFrame) -> pd.Series:
        """Calculate hold times for positions (simplified)"""
        if df.empty:
            return pd.Series()

        # Group by market and calculate time between trades
        hold_times = []

        for market_id in df['market_id'].unique():
            market_df = df[df['market_id'] == market_id].sort_values('timestamp')

            if len(market_df) < 2:
                continue

            # Simple approach: time between consecutive trades
            time_diffs = market_df['timestamp'].diff().dropna()
            hours = time_diffs.dt.total_seconds() / 3600
            hold_times.extend(hours.tolist())

        return pd.Series(hold_times) if hold_times else pd.Series()

    def _split_hold_times(self, df: pd.DataFrame):
        """Split hold times into winning and losing"""
        winning_times = []
        losing_times = []

        for market_id in df['market_id'].unique():
            market_df = df[df['market_id'] == market_id].sort_values('timestamp')

            if len(market_df) < 2:
                continue

            for i in range(1, len(market_df)):
                time_diff = (market_df.iloc[i]['timestamp'] - market_df.iloc[i-1]['timestamp']).total_seconds() / 3600
                pnl = market_df.iloc[i]['pnl']

                if pnl > 0:
                    winning_times.append(time_diff)
                elif pnl < 0:
                    losing_times.append(time_diff)

        return pd.Series(winning_times), pd.Series(losing_times)

    def _calculate_entry_timing_score(self, df: pd.DataFrame) -> float:
        """
        Calculate entry timing quality (0-100).

        Good entry = buying below market mid, selling above market mid
        """
        if df.empty:
            return 50.0

        # For each trade, check if price was favorable
        buy_trades = df[df['side'] == 'BUY']
        sell_trades = df[df['side'] == 'SELL']

        scores = []

        # BUY trades: lower price is better
        if not buy_trades.empty:
            buy_prices = buy_trades['price']
            # Normalize: 0.3-0.7 price range maps to 30-70 score, inverted for buys
            buy_scores = 100 - (buy_prices * 100)
            scores.extend(buy_scores.tolist())

        # SELL trades: higher price is better
        if not sell_trades.empty:
            sell_prices = sell_trades['price']
            sell_scores = sell_prices * 100
            scores.extend(sell_scores.tolist())

        return np.mean(scores) if scores else 50.0

    def _calculate_exit_timing_score(self, df: pd.DataFrame) -> float:
        """
        Calculate exit timing quality (0-100).

        Good exit = closing positions near optimal price
        """
        if df.empty:
            return 50.0

        pnl_series = self.calculate_pnl_series(df)

        # Score based on win rate and average PnL
        win_rate = (pnl_series > 0).mean()
        avg_pnl = pnl_series.mean()
        max_possible_pnl = df['value'].mean() * 0.5  # Assume max 50% gain

        pnl_score = self.safe_divide(avg_pnl, max_possible_pnl) * 50 + 50
        pnl_score = max(0, min(100, pnl_score))

        win_rate_score = win_rate * 100

        return (pnl_score + win_rate_score) / 2
