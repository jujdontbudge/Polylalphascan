"""Group L: Market Intelligence Metrics (IDs 101-104) [NEW IN v2.0]"""
from typing import Dict
import pandas as pd
import numpy as np

from .base_group import BaseMetricGroup, WalletData


class GroupLMarketIntel(BaseMetricGroup):
    """Calculate market intelligence metrics"""

    GROUP_ID = "L"
    METRIC_IDS = list(range(101, 105))  # 101-104

    def calculate(self, data: WalletData) -> Dict[int, float]:
        """Calculate all market intelligence metrics"""
        results = {}
        df = data.trades_df

        if df.empty:
            return {mid: 0.0 for mid in self.METRIC_IDS}

        # Metric 101: Market Timing Accuracy (%)
        results[101] = self._calculate_timing_accuracy(df)

        # Metric 102: Category Expertise Score
        results[102] = self._calculate_category_expertise(df)

        # Metric 103: Price Prediction Skill
        results[103] = self._calculate_prediction_skill(df)

        # Metric 104: Event Outcome Accuracy (%)
        results[104] = self._calculate_event_accuracy(df)

        return results

    def _calculate_timing_accuracy(self, df: pd.DataFrame) -> float:
        """
        Calculate market timing accuracy.

        Correct directional calls on price movements.

        Uses pandas groupby + shift for O(n) performance instead of O(n²) nested loops.
        Expected 10-30x speedup.
        """
        if len(df) < 5:
            return 50.0

        # Sort by market and timestamp for correct groupby shift
        df_sorted = df.sort_values(['market_id', 'timestamp']).copy()

        # Use groupby + shift to get next price within each market
        df_sorted['next_price'] = df_sorted.groupby('market_id')['price'].shift(-1)

        # Calculate if price went up
        df_sorted['price_went_up'] = df_sorted['next_price'] > df_sorted['price']

        # BUY is correct if next price is higher
        buy_correct = (df_sorted['side'] == 'BUY') & df_sorted['price_went_up']

        # SELL is correct if next price is lower
        sell_correct = (df_sorted['side'] == 'SELL') & ~df_sorted['price_went_up']

        # Only count trades where we have a next trade to compare (valid)
        valid = df_sorted['next_price'].notna()

        correct_count = ((buy_correct | sell_correct) & valid).sum()
        total_count = valid.sum()

        return self.safe_divide(correct_count, total_count) * 100 if total_count > 0 else 50

    def _calculate_category_expertise(self, df: pd.DataFrame) -> float:
        """
        Calculate expertise in market categories.

        Higher concentration + good performance = expertise
        """
        if df.empty:
            return 0.0

        pnl_series = self.calculate_pnl_series(df)
        df_with_pnl = df.copy()
        df_with_pnl['pnl'] = pnl_series

        # Calculate PnL per market
        market_performance = df_with_pnl.groupby('market_id').agg({
            'pnl': 'sum',
            'value': 'sum',
        })

        if market_performance.empty:
            return 0.0

        # Best performing market
        best_market = market_performance['pnl'].idxmax()
        best_pnl = market_performance.loc[best_market, 'pnl']
        best_volume = market_performance.loc[best_market, 'value']

        # Expertise score: high ROI in concentrated market
        roi = self.safe_divide(best_pnl, best_volume)
        volume_concentration = self.safe_divide(best_volume, df['value'].sum())

        # Score: ROI * concentration, normalized
        score = roi * volume_concentration * 10000  # Scale up
        return min(100, max(0, score + 50))  # Center around 50

    def _calculate_prediction_skill(self, df: pd.DataFrame) -> float:
        """
        Calculate price prediction skill.

        Accuracy of implied price targets from trade prices.
        """
        if len(df) < 3:
            return 50.0

        predictions_correct = 0
        total_predictions = 0

        for market_id in df['market_id'].unique():
            market_df = df[df['market_id'] == market_id].sort_values('timestamp')

            if len(market_df) < 3:
                continue

            # Each trade implies a prediction about fair value
            for i in range(len(market_df) - 2):
                trade = market_df.iloc[i]
                future_trade = market_df.iloc[i + 2]  # Look 2 trades ahead

                # BUY implies fair value is above current price
                # SELL implies fair value is below current price
                if trade['side'] == 'BUY':
                    # Correct if future price is higher
                    if future_trade['price'] > trade['price']:
                        predictions_correct += 1
                else:
                    # Correct if future price is lower
                    if future_trade['price'] < trade['price']:
                        predictions_correct += 1

                total_predictions += 1

        return self.safe_divide(predictions_correct, total_predictions) * 100 if total_predictions > 0 else 50

    def _calculate_event_accuracy(self, df: pd.DataFrame) -> float:
        """
        Calculate event outcome accuracy.

        For resolved markets, check if position matched outcome.
        This is simplified without actual resolution data.
        """
        if df.empty:
            return 50.0

        # Without actual resolution data, estimate from final prices
        # Trades near 0 or 1 suggest conviction about outcome
        high_conviction_trades = df[(df['price'] < 0.2) | (df['price'] > 0.8)]

        if high_conviction_trades.empty:
            return 50.0

        # Calculate PnL on high conviction trades
        pnl_series = self.calculate_pnl_series(high_conviction_trades)

        if pnl_series.empty:
            return 50.0

        # Win rate on high conviction = event accuracy proxy
        win_rate = (pnl_series > 0).mean()
        return win_rate * 100
