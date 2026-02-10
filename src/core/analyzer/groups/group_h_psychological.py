"""Group H: Psychological Metrics (IDs 67-74)"""
from typing import Dict
import pandas as pd
import numpy as np

from .base_group import BaseMetricGroup, WalletData


class GroupHPsychological(BaseMetricGroup):
    """Calculate psychological/behavioral metrics"""

    GROUP_ID = "H"
    METRIC_IDS = list(range(67, 75))  # 67-74

    def calculate(self, data: WalletData) -> Dict[int, float]:
        """Calculate all psychological metrics"""
        results = {}
        df = data.trades_df

        if df.empty:
            return {mid: 0.0 for mid in self.METRIC_IDS}

        pnl_series = self.calculate_pnl_series(df)
        df_with_pnl = df.copy()
        df_with_pnl['pnl'] = pnl_series

        # Metric 67: Disposition Effect
        # Tendency to hold losers longer, sell winners early
        results[67] = self._calculate_disposition_effect(df_with_pnl)

        # Metric 68: Revenge Trading Score
        # Trading frequency increase after losses
        results[68] = self._calculate_revenge_trading(df_with_pnl)

        # Metric 69: Overconfidence Index
        # Position size increase after wins
        results[69] = self._calculate_overconfidence(df_with_pnl)

        # Metric 70: Loss Aversion Ratio
        # Pain of loss vs pleasure of gain
        results[70] = self._calculate_loss_aversion(df_with_pnl)

        # Metric 71: FOMO Index
        # Chasing trades after missing moves
        results[71] = self._calculate_fomo_index(df)

        # Metric 72: Panic Sell Frequency
        # Exits during high volatility periods
        results[72] = self._calculate_panic_sells(df)

        # Metric 73: Patience Score
        # Ability to wait for setups
        results[73] = self._calculate_patience_score(df)

        # Metric 74: Discipline Score
        # Adherence to position sizing rules
        results[74] = self._calculate_discipline_score(df)

        return results

    def _calculate_disposition_effect(self, df: pd.DataFrame) -> float:
        """
        Calculate disposition effect.

        Disposition effect = holding losers longer than winners
        Higher value = more disposition effect (bad)
        """
        if len(df) < 4:
            return 50.0

        df_sorted = df.sort_values('timestamp')

        # For each market, track hold times for winners vs losers
        winner_times = []
        loser_times = []

        for market_id in df['market_id'].unique():
            market_df = df_sorted[df_sorted['market_id'] == market_id]

            if len(market_df) < 2:
                continue

            for i in range(1, len(market_df)):
                time_diff = (market_df.iloc[i]['timestamp'] - market_df.iloc[i-1]['timestamp']).total_seconds() / 3600
                pnl = market_df.iloc[i]['pnl']

                if pnl > 0:
                    winner_times.append(time_diff)
                elif pnl < 0:
                    loser_times.append(time_diff)

        if not winner_times or not loser_times:
            return 50.0

        avg_winner_time = np.mean(winner_times)
        avg_loser_time = np.mean(loser_times)

        # Higher loser time / winner time = more disposition effect
        ratio = self.safe_divide(avg_loser_time - avg_winner_time, avg_winner_time + avg_loser_time)
        return (ratio + 1) * 50  # Scale to 0-100

    def _calculate_revenge_trading(self, df: pd.DataFrame) -> float:
        """
        Calculate revenge trading tendency.

        Higher trade frequency after losses = more revenge trading
        """
        if len(df) < 5:
            return 0.0

        df_sorted = df.sort_values('timestamp')
        df_sorted['time_to_next'] = df_sorted['timestamp'].diff().shift(-1).dt.total_seconds()

        # Average time after wins vs losses
        after_wins = df_sorted[df_sorted['pnl'] > 0]['time_to_next'].dropna()
        after_losses = df_sorted[df_sorted['pnl'] < 0]['time_to_next'].dropna()

        if after_wins.empty or after_losses.empty:
            return 0.0

        avg_after_win = after_wins.mean()
        avg_after_loss = after_losses.mean()

        # Lower time after loss = revenge trading
        if avg_after_loss < avg_after_win:
            ratio = self.safe_divide(avg_after_win - avg_after_loss, avg_after_win)
            return min(100, ratio * 100)
        return 0.0

    def _calculate_overconfidence(self, df: pd.DataFrame) -> float:
        """
        Calculate overconfidence index.

        Position size increase after wins = overconfidence
        """
        if len(df) < 5:
            return 0.0

        df_sorted = df.sort_values('timestamp')
        df_sorted['next_size'] = df_sorted['value'].shift(-1)

        # Average size after wins vs baseline
        after_wins = df_sorted[df_sorted['pnl'] > 0]['next_size'].dropna()
        baseline = df_sorted['value'].mean()

        if after_wins.empty:
            return 0.0

        avg_after_win = after_wins.mean()

        if avg_after_win > baseline:
            ratio = self.safe_divide(avg_after_win - baseline, baseline)
            return min(100, ratio * 100)
        return 0.0

    def _calculate_loss_aversion(self, df: pd.DataFrame) -> float:
        """
        Calculate loss aversion ratio.

        Loss aversion = behavior change after loss / behavior change after win
        """
        if len(df) < 5:
            return 1.0  # Neutral

        df_sorted = df.sort_values('timestamp')

        # Measure size changes after wins vs losses
        df_sorted['size_change'] = df_sorted['value'].diff().shift(-1).abs()

        after_wins = df_sorted[df_sorted['pnl'] > 0]['size_change'].dropna()
        after_losses = df_sorted[df_sorted['pnl'] < 0]['size_change'].dropna()

        if after_wins.empty or after_losses.empty:
            return 1.0

        return self.safe_divide(after_losses.mean(), after_wins.mean(), default=1.0)

    def _calculate_fomo_index(self, df: pd.DataFrame) -> float:
        """
        Calculate FOMO index.

        Buying after large price increases = FOMO
        """
        if len(df) < 3:
            return 0.0

        df_sorted = df.sort_values('timestamp')
        fomo_count = 0
        total_count = 0

        for market_id in df['market_id'].unique():
            market_df = df_sorted[df_sorted['market_id'] == market_id].reset_index()

            for i in range(1, len(market_df)):
                if market_df.iloc[i]['side'] != 'BUY':
                    continue

                prev_price = market_df.iloc[i-1]['price']
                curr_price = market_df.iloc[i]['price']

                # Large price increase before buy
                if curr_price > prev_price * 1.1:  # 10%+ increase
                    fomo_count += 1
                total_count += 1

        return self.safe_divide(fomo_count, total_count) * 100 if total_count > 0 else 0

    def _calculate_panic_sells(self, df: pd.DataFrame) -> float:
        """
        Calculate panic sell frequency.

        Selling after large price drops = panic
        """
        if len(df) < 3:
            return 0.0

        df_sorted = df.sort_values('timestamp')
        panic_count = 0
        total_sells = 0

        for market_id in df['market_id'].unique():
            market_df = df_sorted[df_sorted['market_id'] == market_id].reset_index()

            for i in range(1, len(market_df)):
                if market_df.iloc[i]['side'] != 'SELL':
                    continue

                total_sells += 1
                prev_price = market_df.iloc[i-1]['price']
                curr_price = market_df.iloc[i]['price']

                # Large price drop before sell
                if curr_price < prev_price * 0.9:  # 10%+ drop
                    panic_count += 1

        return self.safe_divide(panic_count, total_sells) * 100 if total_sells > 0 else 0

    def _calculate_patience_score(self, df: pd.DataFrame) -> float:
        """
        Calculate patience score.

        Higher = more selective trading
        """
        if df.empty:
            return 50.0

        # Measure by trade frequency relative to market activity
        df_sorted = df.sort_values('timestamp')
        time_span = (df_sorted['timestamp'].max() - df_sorted['timestamp'].min()).days + 1

        trades_per_day = len(df) / time_span

        # Fewer trades = more patience (up to a point)
        # 1-3 trades/day = high patience, 10+ = low
        if trades_per_day <= 1:
            return 90
        elif trades_per_day <= 3:
            return 75
        elif trades_per_day <= 5:
            return 60
        elif trades_per_day <= 10:
            return 40
        else:
            return max(20, 100 - trades_per_day * 5)

    def _calculate_discipline_score(self, df: pd.DataFrame) -> float:
        """
        Calculate discipline score.

        Consistent position sizing = high discipline
        """
        if len(df) < 3:
            return 50.0

        # Lower coefficient of variation = more disciplined
        cv = self.safe_divide(df['value'].std(), df['value'].mean())

        # CV of 0 = 100, CV of 1 = 50, CV of 2 = 0
        score = max(0, 100 - cv * 50)
        return min(100, score)
