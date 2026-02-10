"""Group G: Strategy Signature Metrics (IDs 58-66)"""
from typing import Dict
import pandas as pd
import numpy as np

from .base_group import BaseMetricGroup, WalletData


class GroupGStrategy(BaseMetricGroup):
    """Calculate strategy signature metrics"""

    GROUP_ID = "G"
    METRIC_IDS = list(range(58, 67))  # 58-66

    def calculate(self, data: WalletData) -> Dict[int, float]:
        """Calculate all strategy metrics"""
        results = {}
        df = data.trades_df

        if df.empty:
            return {mid: 0.0 for mid in self.METRIC_IDS}

        buy_volume = df[df['side'] == 'BUY']['value'].sum()
        sell_volume = df[df['side'] == 'SELL']['value'].sum()
        total_volume = buy_volume + sell_volume

        # Metric 58: Buy/Sell Ratio
        results[58] = self.safe_divide(buy_volume, sell_volume, default=1.0)

        # Metric 59: Long/Short Bias (-1 to 1)
        # Positive = long bias, Negative = short bias
        results[59] = self.safe_divide(buy_volume - sell_volume, total_volume)

        # Metric 60: Contrarian Score
        # Measures trading against price trends
        contrarian_score = self._calculate_contrarian_score(df)
        results[60] = contrarian_score

        # Metric 61: Momentum Score
        # Measures trading with price trends
        results[61] = 100 - contrarian_score  # Inverse of contrarian

        # Metric 62: Mean Reversion Score
        # Measures buying after drops, selling after rises
        results[62] = self._calculate_mean_reversion_score(df)

        # Metric 63: Arbitrage Activity (%)
        # Quick round-trip trades
        arb_trades = self._count_arb_trades(df)
        results[63] = self.safe_divide(arb_trades, len(df)) * 100

        # Metric 64: Market Making Ratio (%)
        # Both-side trading in same market
        mm_markets = self._count_market_making_markets(df)
        total_markets = df['market_id'].nunique()
        results[64] = self.safe_divide(mm_markets, total_markets) * 100

        # Metric 65: Event Trading Ratio (%)
        # This would need event data; estimate based on timing
        results[65] = 0  # Placeholder - needs event resolution data

        # Metric 66: Strategy Consistency
        # How consistent is the trading pattern
        results[66] = self._calculate_strategy_consistency(df)

        return results

    def _calculate_contrarian_score(self, df: pd.DataFrame) -> float:
        """Calculate contrarian trading tendency"""
        if len(df) < 2:
            return 50.0

        df_sorted = df.sort_values('timestamp')
        contrarian_count = 0
        total_count = 0

        for market_id in df['market_id'].unique():
            market_df = df_sorted[df_sorted['market_id'] == market_id].reset_index()

            for i in range(1, len(market_df)):
                prev_price = market_df.iloc[i-1]['price']
                curr_price = market_df.iloc[i]['price']
                side = market_df.iloc[i]['side']

                price_up = curr_price > prev_price

                # Contrarian: buy when price falling, sell when price rising
                if (side == 'BUY' and not price_up) or (side == 'SELL' and price_up):
                    contrarian_count += 1
                total_count += 1

        return self.safe_divide(contrarian_count, total_count) * 100 if total_count > 0 else 50

    def _calculate_mean_reversion_score(self, df: pd.DataFrame) -> float:
        """Calculate mean reversion trading tendency"""
        if len(df) < 3:
            return 50.0

        # Look for buy after drops and sell after rises
        df_sorted = df.sort_values('timestamp')
        reversion_count = 0
        total_count = 0

        for market_id in df['market_id'].unique():
            market_df = df_sorted[df_sorted['market_id'] == market_id].reset_index()

            if len(market_df) < 3:
                continue

            for i in range(2, len(market_df)):
                price_2back = market_df.iloc[i-2]['price']
                price_1back = market_df.iloc[i-1]['price']
                curr_side = market_df.iloc[i]['side']

                # Recent trend
                recent_up = price_1back > price_2back

                # Mean reversion: buy after down trend, sell after up trend
                if (curr_side == 'BUY' and not recent_up) or (curr_side == 'SELL' and recent_up):
                    reversion_count += 1
                total_count += 1

        return self.safe_divide(reversion_count, total_count) * 100 if total_count > 0 else 50

    def _count_arb_trades(self, df: pd.DataFrame) -> int:
        """Count potential arbitrage trades (quick round trips)"""
        if len(df) < 2:
            return 0

        arb_count = 0
        df_sorted = df.sort_values('timestamp')

        for market_id in df['market_id'].unique():
            market_df = df_sorted[df_sorted['market_id'] == market_id]

            if len(market_df) < 2:
                continue

            for i in range(1, len(market_df)):
                time_diff = (market_df.iloc[i]['timestamp'] - market_df.iloc[i-1]['timestamp']).total_seconds()
                sides_differ = market_df.iloc[i]['side'] != market_df.iloc[i-1]['side']

                # Quick round trip: < 5 minutes, different sides
                if time_diff < 300 and sides_differ:
                    arb_count += 1

        return arb_count

    def _count_market_making_markets(self, df: pd.DataFrame) -> int:
        """Count markets with both buy and sell activity"""
        both_side_markets = 0

        for market_id in df['market_id'].unique():
            market_df = df[df['market_id'] == market_id]
            sides = market_df['side'].unique()

            if 'BUY' in sides and 'SELL' in sides:
                both_side_markets += 1

        return both_side_markets

    def _calculate_strategy_consistency(self, df: pd.DataFrame) -> float:
        """Calculate how consistent the trading strategy is"""
        if len(df) < 5:
            return 50.0

        # Measure consistency in:
        # 1. Trade size variance
        # 2. Time between trades variance
        # 3. Buy/sell ratio variance over time

        # Trade size consistency (lower variance = more consistent)
        size_cv = self.safe_divide(df['value'].std(), df['value'].mean())
        size_score = max(0, 100 - size_cv * 100)

        # Time consistency
        df_sorted = df.sort_values('timestamp')
        time_diffs = df_sorted['timestamp'].diff().dropna().dt.total_seconds()
        if not time_diffs.empty and time_diffs.mean() > 0:
            time_cv = self.safe_divide(time_diffs.std(), time_diffs.mean())
            time_score = max(0, 100 - time_cv * 50)
        else:
            time_score = 50

        return (size_score + time_score) / 2
