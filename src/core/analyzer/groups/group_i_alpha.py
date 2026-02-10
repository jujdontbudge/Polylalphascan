"""Group I: Advanced Alpha Metrics (IDs 75-83)"""
from typing import Dict
import pandas as pd
import numpy as np

from .base_group import BaseMetricGroup, WalletData


class GroupIAlpha(BaseMetricGroup):
    """Calculate advanced alpha metrics"""

    GROUP_ID = "I"
    METRIC_IDS = list(range(75, 84))  # 75-83

    def calculate(self, data: WalletData) -> Dict[int, float]:
        """Calculate all advanced alpha metrics"""
        results = {}
        df = data.trades_df

        if df.empty:
            return {mid: 0.0 for mid in self.METRIC_IDS}

        pnl_series = self.calculate_pnl_series(df)
        returns = pnl_series / df['value']
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

        # Metric 75: Alpha vs Market
        # Simplified: use average return as proxy
        market_return = 0.05  # Assumed market return (5%)
        portfolio_return = returns.mean() * len(returns) if not returns.empty else 0
        results[75] = (portfolio_return - market_return) * 100

        # Metric 76: Information Ratio
        tracking_error = returns.std() if len(returns) > 1 else 0.01
        alpha = portfolio_return - market_return
        results[76] = self.safe_divide(alpha, tracking_error)

        # Metric 77: Treynor Ratio
        risk_free = 0.02  # 2% risk free rate
        beta = self._calculate_beta(returns)
        excess_return = portfolio_return - risk_free
        results[77] = self.safe_divide(excess_return, beta)

        # Metric 78: Beta
        results[78] = beta

        # Metric 79: Edge per Trade (bps)
        avg_trade_size = df['value'].mean()
        expectancy = pnl_series.mean() if not pnl_series.empty else 0
        results[79] = self.safe_divide(expectancy, avg_trade_size) * 10000

        # Metric 80: Hit Rate on High Conviction
        results[80] = self._calculate_high_conviction_hit_rate(df, pnl_series)

        # Metric 81: Asymmetric Payoff Ratio
        results[81] = self._calculate_asymmetric_payoff(returns)

        # Metric 82: Smart Money Indicator
        results[82] = self._calculate_smart_money_indicator(df)

        # Metric 83: Early Mover Advantage
        results[83] = self._calculate_early_mover_advantage(df, pnl_series)

        return results

    def _calculate_beta(self, returns: pd.Series) -> float:
        """
        Calculate beta (market sensitivity).

        Simplified: use volatility as proxy
        """
        if len(returns) < 5:
            return 1.0

        # Without actual market returns, estimate beta from volatility
        vol = returns.std()
        market_vol = 0.02  # Assumed market volatility

        # Beta = correlation * (asset_vol / market_vol)
        # Assume correlation of 0.5 for prediction markets
        beta = 0.5 * (vol / market_vol) if market_vol > 0 else 1.0
        return max(0.1, min(3.0, beta))

    def _calculate_high_conviction_hit_rate(
        self,
        df: pd.DataFrame,
        pnl_series: pd.Series,
    ) -> float:
        """
        Calculate win rate on larger positions.

        High conviction = above-average position size
        """
        if df.empty or pnl_series.empty:
            return 50.0

        avg_size = df['value'].mean()
        df_with_pnl = df.copy()
        df_with_pnl['pnl'] = pnl_series

        # High conviction = size > average
        high_conviction = df_with_pnl[df_with_pnl['value'] > avg_size]

        if high_conviction.empty:
            return 50.0

        wins = (high_conviction['pnl'] > 0).sum()
        total = len(high_conviction)

        return self.safe_divide(wins, total) * 100

    def _calculate_asymmetric_payoff(self, returns: pd.Series) -> float:
        """
        Calculate upside/downside capture ratio.

        Higher = better at capturing gains vs limiting losses
        """
        if returns.empty:
            return 1.0

        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]

        avg_upside = positive_returns.mean() if not positive_returns.empty else 0
        avg_downside = abs(negative_returns.mean()) if not negative_returns.empty else 0.01

        return self.safe_divide(avg_upside, avg_downside, default=1.0)

    def _calculate_smart_money_indicator(self, df: pd.DataFrame) -> float:
        """
        Calculate smart money alignment.

        Measures if trades align with market-moving activity
        """
        if len(df) < 5:
            return 50.0

        # Heuristic: trades at extreme prices (near 0 or 1) that later move toward fair value
        df_sorted = df.sort_values('timestamp')

        smart_trades = 0
        total_trades = 0

        for market_id in df['market_id'].unique():
            market_df = df_sorted[df_sorted['market_id'] == market_id]

            if len(market_df) < 2:
                continue

            # Get price range in this market
            min_price = market_df['price'].min()
            max_price = market_df['price'].max()

            for _, trade in market_df.iterrows():
                total_trades += 1

                # Smart: buying near lows, selling near highs
                is_extreme_low = trade['price'] <= min_price * 1.1
                is_extreme_high = trade['price'] >= max_price * 0.9

                if (trade['side'] == 'BUY' and is_extreme_low) or \
                   (trade['side'] == 'SELL' and is_extreme_high):
                    smart_trades += 1

        return self.safe_divide(smart_trades, total_trades) * 100 if total_trades > 0 else 50

    def _calculate_early_mover_advantage(
        self,
        df: pd.DataFrame,
        pnl_series: pd.Series,
    ) -> float:
        """
        Calculate advantage from early market entry.

        Measures profit from trades early in market lifecycle
        """
        if df.empty or pnl_series.empty:
            return 0.0

        df_with_pnl = df.copy()
        df_with_pnl['pnl'] = pnl_series

        # Group by market and calculate profit from first vs later trades
        early_pnl = 0
        total_pnl = pnl_series.sum()

        for market_id in df['market_id'].unique():
            market_df = df_with_pnl[df_with_pnl['market_id'] == market_id].sort_values('timestamp')

            if len(market_df) < 2:
                continue

            # First 25% of trades in market are "early"
            early_count = max(1, len(market_df) // 4)
            early_trades = market_df.head(early_count)
            early_pnl += early_trades['pnl'].sum()

        if total_pnl == 0:
            return 50.0

        # Percentage of profits from early trades
        early_ratio = self.safe_divide(early_pnl, total_pnl)
        return early_ratio * 100
