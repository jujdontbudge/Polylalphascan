"""Group B: Risk & Stability Metrics (IDs 11-21)"""
from typing import Dict
import pandas as pd
import numpy as np

from .base_group import BaseMetricGroup, WalletData


class GroupBRisk(BaseMetricGroup):
    """Calculate risk and stability metrics"""

    GROUP_ID = "B"
    METRIC_IDS = list(range(11, 22))  # 11-21

    def calculate(self, data: WalletData) -> Dict[int, float]:
        """Calculate all risk metrics"""
        results = {}
        df = data.trades_df

        if df.empty:
            return {mid: 0.0 for mid in self.METRIC_IDS}

        # Calculate returns
        pnl_series = self.calculate_pnl_series(df)
        returns = pnl_series / df['value']
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

        risk_free_rate = 0.0  # Simplified

        # Metric 11: Sharpe Ratio
        mean_return = returns.mean() if not returns.empty else 0
        std_return = returns.std() if len(returns) > 1 else 1
        results[11] = self.safe_divide(mean_return - risk_free_rate, std_return)

        # Metric 12: Sortino Ratio
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std() if len(negative_returns) > 1 else 1
        results[12] = self.safe_divide(mean_return - risk_free_rate, downside_std)

        # Calculate drawdowns
        cumulative = (1 + returns).cumprod() if not returns.empty else pd.Series([1])
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        drawdowns = drawdowns.replace([np.inf, -np.inf], 0).fillna(0)

        # Metric 13: Calmar Ratio
        max_dd = abs(drawdowns.min()) if not drawdowns.empty else 0.01
        annualized_return = mean_return * 252 if mean_return else 0  # Approximate trading days
        results[13] = self.safe_divide(annualized_return, max_dd)

        # Metric 14: Maximum Drawdown (%)
        results[14] = abs(drawdowns.min()) * 100 if not drawdowns.empty else 0

        # Metric 15: Average Drawdown (%)
        dd_periods = drawdowns[drawdowns < 0]
        results[15] = abs(dd_periods.mean()) * 100 if not dd_periods.empty else 0

        # Metric 16: Max Drawdown Duration (days)
        # Simplified: count consecutive negative periods
        if not drawdowns.empty:
            in_dd = (drawdowns < 0).astype(int)
            dd_groups = (in_dd != in_dd.shift()).cumsum()
            dd_lengths = in_dd.groupby(dd_groups).sum()
            results[16] = dd_lengths.max() if not dd_lengths.empty else 0
        else:
            results[16] = 0

        # Metric 17: Return Volatility (%)
        results[17] = std_return * 100 if std_return else 0

        # Metric 18: Downside Volatility (%)
        results[18] = downside_std * 100 if downside_std else 0

        # Metric 19: Value at Risk (5%)
        results[19] = self.percentile(returns, 5) * 100 if not returns.empty else 0

        # Metric 20: Conditional VaR (5%)
        var_5 = self.percentile(returns, 5)
        below_var = returns[returns <= var_5]
        results[20] = below_var.mean() * 100 if not below_var.empty else 0

        # Metric 21: Recovery Factor
        total_pnl = pnl_series.sum() if not pnl_series.empty else 0
        max_dd_usd = abs(drawdowns.min() * df['value'].sum()) if not drawdowns.empty else 1
        results[21] = self.safe_divide(total_pnl, max_dd_usd)

        return results
