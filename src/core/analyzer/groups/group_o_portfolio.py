"""Group O: Portfolio-Level Metrics (IDs 114-117) [NEW IN v2.0]"""
from typing import Dict
import pandas as pd
import numpy as np

from .base_group import BaseMetricGroup, WalletData


class GroupOPortfolio(BaseMetricGroup):
    """Calculate portfolio-level metrics"""

    GROUP_ID = "O"
    METRIC_IDS = list(range(114, 118))  # 114-117

    def calculate(self, data: WalletData) -> Dict[int, float]:
        """Calculate all portfolio-level metrics"""
        results = {}
        df = data.trades_df
        positions_df = data.positions_df

        if df.empty:
            return {mid: 0.0 for mid in self.METRIC_IDS}

        # Metric 114: Portfolio Heat Score
        results[114] = self._calculate_portfolio_heat(df, positions_df)

        # Metric 115: Correlation Exposure
        results[115] = self._calculate_correlation_exposure(df)

        # Metric 116: Sector Balance Score
        results[116] = self._calculate_sector_balance(df)

        # Metric 117: Net Exposure (%)
        results[117] = self._calculate_net_exposure(df, positions_df)

        return results

    def _calculate_portfolio_heat(
        self,
        df: pd.DataFrame,
        positions_df: pd.DataFrame,
    ) -> float:
        """
        Calculate portfolio heat (risk level).

        Higher = more risk concentrated in current positions.
        """
        if positions_df.empty:
            # Estimate from recent trades
            if df.empty:
                return 0.0

            # Recent activity as proxy for heat
            recent = df.tail(10)
            total_recent_value = recent['value'].sum()
            total_value = df['value'].sum()

            recency_ratio = self.safe_divide(total_recent_value, total_value)
            return recency_ratio * 100

        # Calculate from positions
        total_position_value = positions_df['value'].sum()
        max_position_value = positions_df['value'].max()

        # Heat factors:
        # 1. Total exposure
        # 2. Concentration in single position
        # 3. Positions with high current risk

        concentration = self.safe_divide(max_position_value, total_position_value)

        # Positions with price near 0.5 have highest risk (uncertainty)
        risk_positions = positions_df[
            (positions_df['current_price'] > 0.3) &
            (positions_df['current_price'] < 0.7)
        ]
        risk_value = risk_positions['value'].sum() if not risk_positions.empty else 0
        risk_ratio = self.safe_divide(risk_value, total_position_value)

        heat_score = (concentration * 50 + risk_ratio * 50)
        return min(100, heat_score)

    def _calculate_correlation_exposure(self, df: pd.DataFrame) -> float:
        """
        Calculate portfolio correlation to market.

        Simplified beta calculation.
        """
        if len(df) < 10:
            return 1.0  # Neutral

        # Group by time period and calculate returns
        df_copy = df.copy()
        df_copy['date'] = df_copy['timestamp'].dt.date

        daily_volume = df_copy.groupby('date')['value'].sum()

        if len(daily_volume) < 5:
            return 1.0

        # Use volume trend as proxy for market correlation
        # Increasing volume when market is active = high correlation
        volume_changes = daily_volume.pct_change().dropna()

        if volume_changes.empty:
            return 1.0

        # Proxy beta: standard deviation of volume changes
        vol_std = volume_changes.std()

        # Scale to typical beta range (0.5 - 2.0)
        beta = 0.5 + vol_std * 3
        return min(2.0, max(0.5, beta))

    def _calculate_sector_balance(self, df: pd.DataFrame) -> float:
        """
        Calculate sector/category balance.

        Higher = more diversified across market categories.
        """
        if df.empty:
            return 0.0

        # Use market_id as proxy for sector
        # (Real implementation would need market category data)
        market_volumes = df.groupby('market_id')['value'].sum()

        if market_volumes.empty:
            return 0.0

        # Calculate Gini coefficient for concentration
        # Lower Gini = more balanced
        weights = market_volumes / market_volumes.sum()
        n = len(weights)

        if n <= 1:
            return 100.0  # Single market = perfectly concentrated

        # Gini formula
        sorted_weights = np.sort(weights)
        cumsum = np.cumsum(sorted_weights)
        gini = 1 - 2 * cumsum.sum() / (n * sorted_weights.sum()) + 1 / n

        # Convert to balance score (1 - Gini) * 100
        balance_score = (1 - gini) * 100
        return balance_score

    def _calculate_net_exposure(
        self,
        df: pd.DataFrame,
        positions_df: pd.DataFrame,
    ) -> float:
        """
        Calculate net directional exposure.

        | long_value - short_value | / total_value * 100
        """
        if positions_df.empty:
            # Estimate from trade history
            if df.empty:
                return 0.0

            buy_volume = df[df['side'] == 'BUY']['value'].sum()
            sell_volume = df[df['side'] == 'SELL']['value'].sum()
            total_volume = buy_volume + sell_volume

            if total_volume == 0:
                return 0.0

            net_exposure = abs(buy_volume - sell_volume) / total_volume * 100
            return net_exposure

        # Calculate from positions
        # Assuming: positions with outcome_index 0 = "Yes" (long), 1 = "No" (short)
        # This is a simplification
        long_positions = positions_df[positions_df['outcome_index'] == 0]
        short_positions = positions_df[positions_df['outcome_index'] == 1]

        long_value = long_positions['value'].sum() if not long_positions.empty else 0
        short_value = short_positions['value'].sum() if not short_positions.empty else 0
        total_value = long_value + short_value

        if total_value == 0:
            return 0.0

        net_exposure = abs(long_value - short_value) / total_value * 100
        return net_exposure
