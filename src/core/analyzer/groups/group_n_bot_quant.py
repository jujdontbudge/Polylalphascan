"""Group N: Bot/Quant Confirmation Metrics (IDs 109-113) [NEW IN v2.0]"""
import logging
from typing import Dict
from collections import Counter
import pandas as pd
import numpy as np
from scipy import stats

from .base_group import (
    BaseMetricGroup,
    WalletData,
    NUMBA_AVAILABLE,
    calculate_interval_entropy_numba,
)

logger = logging.getLogger(__name__)


class GroupNBotQuant(BaseMetricGroup):
    """
    Calculate bot/quant confirmation metrics.

    Key metrics for identifying algorithmic traders:
    - 109: Trade Interval Entropy (low = bot)
    - 110: Round Number Avoidance (high = bot)
    - 111: Latency Consistency (low variance = bot)
    - 112: Multi-Market Simultaneity (high = bot)
    - 113: Session Pattern Regularity (high = cron job)
    """

    GROUP_ID = "N"
    METRIC_IDS = list(range(109, 114))  # 109-113

    def calculate(self, data: WalletData) -> Dict[int, float]:
        """Calculate all bot/quant confirmation metrics"""
        results = {}
        df = data.trades_df

        if df.empty:
            return {mid: 0.0 for mid in self.METRIC_IDS}

        # Metric 109: Trade Interval Entropy (low = bot)
        results[109] = self._calculate_interval_entropy(df)

        # Metric 110: Round Number Avoidance (%) (high = bot)
        results[110] = self._calculate_round_number_avoidance(df)

        # Metric 111: Latency Consistency Score (low variance = bot)
        results[111] = self._calculate_latency_consistency(df)

        # Metric 112: Multi-Market Simultaneity (high = bot)
        results[112] = self._calculate_simultaneity(df)

        # Metric 113: Session Pattern Regularity (high = cron job)
        results[113] = self._calculate_session_regularity(df)

        return results

    def _calculate_interval_entropy(self, df: pd.DataFrame) -> float:
        """
        Calculate entropy of trade intervals.

        Low entropy = regular/predictable intervals = likely bot
        High entropy = random intervals = likely human

        Uses Numba JIT compilation when available for 5-10x speedup.
        """
        if len(df) < 5:
            return 50.0  # Neutral

        df_sorted = df.sort_values('timestamp')
        intervals = df_sorted['timestamp'].diff().dropna().dt.total_seconds()

        if intervals.empty or len(intervals) < 3:
            return 50.0

        try:
            num_bins = max(2, min(20, len(intervals) // 2))

            if NUMBA_AVAILABLE:
                # Use JIT-compiled version
                return calculate_interval_entropy_numba(
                    intervals.values.astype(np.float64),
                    num_bins
                )

            # Fallback: scipy-based calculation
            hist, _ = np.histogram(intervals, bins=num_bins)
            hist = hist[hist > 0]  # Remove zeros

            if len(hist) < 2:
                # All intervals in 1 bin = perfectly regular = strong bot signal
                return 0.0

            # Normalize to probabilities
            probs = hist / hist.sum()

            # Calculate entropy
            entropy = stats.entropy(probs)

            # Normalize: max entropy depends on bins
            max_entropy = np.log(len(hist))
            if max_entropy == 0:
                return 0.0
            normalized_entropy = self.safe_divide(entropy, max_entropy) * 100

            return normalized_entropy
        except Exception as e:
            logger.warning(f"Interval entropy calculation failed: {e}")
            return 50.0

    def _calculate_round_number_avoidance(self, df: pd.DataFrame) -> float:
        """
        Calculate percentage of trades avoiding round numbers.

        Bots often avoid round prices (0.50, 0.25, etc.) to avoid detection
        or use precise calculations.

        Round numbers are based on common fractions of the price range,
        adapting to both Polymarket (0-1) and other ranges.

        Uses NumPy broadcasting for O(n) performance instead of O(n×m) nested loop.
        Expected 10-50x speedup.
        """
        if df.empty:
            return 50.0

        prices = df['price'].values
        min_price = prices.min()
        max_price = prices.max()
        price_range = max_price - min_price

        # Generate round numbers as fractions of the range
        fractions = np.array([0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0])
        round_numbers = min_price + fractions * price_range

        # Adaptive tolerance proportional to price range
        tolerance = 0.005 * (price_range + 1)

        # Vectorized: compute differences matrix [n_prices x n_round_numbers]
        # Then check if any difference is below tolerance for each price
        differences = np.abs(prices[:, np.newaxis] - round_numbers)
        is_round = np.any(differences < tolerance, axis=1)

        # Count non-round prices
        non_round_count = np.sum(~is_round)

        return self.safe_divide(non_round_count, len(prices)) * 100

    def _calculate_latency_consistency(self, df: pd.DataFrame) -> float:
        """
        Calculate latency consistency.

        Low variance in execution timing = likely bot
        Returns coefficient of variation (lower = more consistent)
        """
        if len(df) < 5:
            return 50.0

        df_sorted = df.sort_values('timestamp')

        # Calculate time between trades
        intervals = df_sorted['timestamp'].diff().dropna().dt.total_seconds()

        if intervals.empty:
            return 50.0

        # Coefficient of variation
        cv = self.safe_divide(intervals.std(), intervals.mean())

        # Normalize: CV < 0.5 = very consistent (bot), CV > 2 = very variable (human)
        # Lower score = more bot-like
        return min(100, cv * 50)

    def _calculate_simultaneity(self, df: pd.DataFrame) -> float:
        """
        Calculate multi-market simultaneity using an optimized sliding window with Counter.

        Count of trades in different markets within short time windows.
        High simultaneity = likely bot (humans can't trade multiple markets simultaneously)

        Uses O(n) sliding window with Counter for efficient market tracking,
        avoiding nested loops. Expected 5-20x speedup.
        """
        if len(df) < 3:
            return 0

        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        timestamps = df_sorted['timestamp'].values
        market_ids = df_sorted['market_id'].values

        time_window_ns = np.timedelta64(10, 's')  # 10 seconds in nanoseconds
        simultaneous_count = 0
        left = 0

        # Counter tracks markets currently in the window
        market_counter: Counter = Counter()
        window_size = 0  # Total trades in window (excluding current)

        for right in range(len(df_sorted)):
            current_market = market_ids[right]

            # Shrink window: remove trades that are too old
            while left < right and (timestamps[right] - timestamps[left]) > time_window_ns:
                old_market = market_ids[left]
                market_counter[old_market] -= 1
                if market_counter[old_market] == 0:
                    del market_counter[old_market]
                window_size -= 1
                left += 1

            # Count trades in window that are in DIFFERENT markets
            # Total trades in window minus trades in same market as current
            same_market_in_window = market_counter.get(current_market, 0)
            different_market_trades = window_size - same_market_in_window
            simultaneous_count += different_market_trades

            # Add current trade to window for next iteration
            market_counter[current_market] += 1
            window_size += 1

        # Cap to prevent extreme values skewing the score
        return min(simultaneous_count, len(df_sorted) * 5)

    def _calculate_session_regularity(self, df: pd.DataFrame) -> float:
        """
        Calculate session pattern regularity.

        High regularity = likely cron job or scheduled bot
        Measures consistency of trading hours/patterns
        """
        if len(df) < 10:
            return 50.0

        df_sorted = df.copy()
        df_sorted['hour'] = df_sorted['timestamp'].dt.hour
        df_sorted['day_of_week'] = df_sorted['timestamp'].dt.dayofweek
        df_sorted['minute'] = df_sorted['timestamp'].dt.minute

        scores = []

        # 1. Hour consistency
        hour_counts = df_sorted['hour'].value_counts()
        if len(hour_counts) > 0:
            # If trades concentrated in few hours = more regular
            top_hours_pct = hour_counts.head(3).sum() / len(df_sorted)
            scores.append(top_hours_pct * 100)

        # 2. Minute patterns (cron jobs often run at specific minutes)
        minute_counts = df_sorted['minute'].value_counts()
        if len(minute_counts) > 0:
            # If trades at same minute(s) = cron-like
            top_minutes_pct = minute_counts.head(5).sum() / len(df_sorted)
            scores.append(top_minutes_pct * 100)

        # 3. Day of week consistency
        dow_counts = df_sorted['day_of_week'].value_counts()
        if len(dow_counts) > 0:
            # Consistent across days = automated
            dow_std = dow_counts.std()
            dow_mean = dow_counts.mean()
            dow_cv = self.safe_divide(dow_std, dow_mean)
            # Lower CV = more regular
            scores.append(max(0, 100 - dow_cv * 100))

        return np.mean(scores) if scores else 50


def calculate_quant_score(metrics: Dict[int, float]) -> float:
    """
    Calculate overall quant/bot score from Group N metrics.

    Score >= 80: "bot"
    Score >= 50: "quant"
    Score <= 20: "human"

    Args:
        metrics: Dict of metric_id to value from Group N (109-113)

    Returns:
        Quant score 0-100
    """
    # Weight each metric for overall bot detection
    weights = {
        109: -0.2,  # Entropy (lower = more bot, so negative weight)
        110: 0.25,  # Round number avoidance (higher = more bot)
        111: -0.2,  # Latency consistency (lower = more bot)
        112: 0.2,   # Simultaneity (higher = more bot)
        113: 0.15,  # Session regularity (higher = more bot)
    }

    score = 50  # Start at neutral

    for metric_id, weight in weights.items():
        if metric_id in metrics:
            value = metrics[metric_id]

            if weight < 0:
                # Inverse metrics (lower = more bot)
                contribution = (100 - value) * abs(weight)
            else:
                # Direct metrics (higher = more bot)
                contribution = value * weight

            score += contribution - 50 * abs(weight)  # Center around 50

    return max(0, min(100, score))
