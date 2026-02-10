"""Group C: Win/Loss Dynamics Metrics (IDs 22-32)"""
from typing import Dict
import pandas as pd
import numpy as np

from .base_group import (
    BaseMetricGroup,
    WalletData,
    NUMBA_AVAILABLE,
    calculate_max_streak_numba,
    calculate_all_streaks_numba,
)


class GroupCWinLoss(BaseMetricGroup):
    """Calculate win/loss dynamics metrics"""

    GROUP_ID = "C"
    METRIC_IDS = list(range(22, 33))  # 22-32

    def calculate(self, data: WalletData) -> Dict[int, float]:
        """Calculate all win/loss metrics"""
        results = {}
        df = data.trades_df

        if df.empty:
            return {mid: 0.0 for mid in self.METRIC_IDS}

        # Calculate PnL for each trade
        pnl_series = self.calculate_pnl_series(df)

        total_trades = len(pnl_series)
        winning_trades = pnl_series[pnl_series > 0]
        losing_trades = pnl_series[pnl_series < 0]

        # Metric 22: Win Rate (%)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        results[22] = win_rate * 100

        # Metric 23: Loss Rate (%)
        loss_rate = len(losing_trades) / total_trades if total_trades > 0 else 0
        results[23] = loss_rate * 100

        # Metric 24: Average Win (USD)
        results[24] = winning_trades.mean() if not winning_trades.empty else 0

        # Metric 25: Average Loss (USD)
        results[25] = abs(losing_trades.mean()) if not losing_trades.empty else 0

        # Metric 26: Win/Loss Ratio
        avg_win = winning_trades.mean() if not winning_trades.empty else 0
        avg_loss = abs(losing_trades.mean()) if not losing_trades.empty else 1
        results[26] = self.safe_divide(avg_win, avg_loss)

        # Calculate streaks
        win_loss_sequence = (pnl_series > 0).astype(int)

        # Metric 27: Max Consecutive Wins
        results[27] = self._max_consecutive(win_loss_sequence, 1)

        # Metric 28: Max Consecutive Losses
        results[28] = self._max_consecutive(win_loss_sequence, 0)

        # Metric 29: Average Streak Length
        streaks = self._get_all_streaks(win_loss_sequence)
        results[29] = np.mean(streaks) if streaks else 0

        # Metric 30: Trade Expectancy
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        results[30] = expectancy

        # Metric 31: Kelly Criterion (%)
        win_loss_ratio = results[26]
        if win_loss_ratio > 0:
            kelly = (win_rate - (1 - win_rate) / win_loss_ratio) * 100
            results[31] = max(0, min(kelly, 100))  # Cap between 0-100%
        else:
            results[31] = 0

        # Metric 32: Break-Even Win Rate (%)
        if win_loss_ratio > 0:
            break_even = (1 / (1 + win_loss_ratio)) * 100
            results[32] = break_even
        else:
            results[32] = 50

        return results

    def _max_consecutive(self, sequence: pd.Series, value: int) -> int:
        """
        Calculate maximum consecutive occurrences of a value.

        Uses Numba JIT compilation when available for 5-10x speedup.
        """
        if sequence.empty:
            return 0

        if NUMBA_AVAILABLE:
            # Use JIT-compiled version
            return calculate_max_streak_numba(sequence.values.astype(np.int64), value)

        # Fallback: pure Python
        max_streak = 0
        current_streak = 0

        for v in sequence:
            if v == value:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    def _get_all_streaks(self, sequence: pd.Series) -> list:
        """
        Get all streak lengths.

        Uses Numba JIT compilation when available for 5-10x speedup.
        """
        if sequence.empty:
            return []

        if NUMBA_AVAILABLE:
            # Use JIT-compiled version
            streaks_arr, count = calculate_all_streaks_numba(sequence.values.astype(np.int64))
            return streaks_arr[:count].tolist()

        # Fallback: pure Python
        streaks = []
        current_streak = 1

        for i in range(1, len(sequence)):
            if sequence.iloc[i] == sequence.iloc[i-1]:
                current_streak += 1
            else:
                streaks.append(current_streak)
                current_streak = 1

        streaks.append(current_streak)
        return streaks
