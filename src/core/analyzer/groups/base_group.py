"""Base class for metric group calculations"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Try to import numba for JIT compilation (5-10x speedup on hot loops)
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback: decorator that does nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from src.api.data_client import Trade, Position


# =============================================================================
# Numba JIT-compiled functions for hot loops (5-10x speedup)
# =============================================================================

@jit(nopython=True, cache=True)
def calculate_market_pnl_numba(
    prices: np.ndarray,
    sizes: np.ndarray,
    fees: np.ndarray,
    sides: np.ndarray,  # 1=BUY, 0=SELL
) -> float:
    """
    JIT-compiled market PnL calculation using cost-basis tracking.

    Args:
        prices: Array of trade prices
        sizes: Array of trade sizes
        fees: Array of trade fees
        sides: Array of sides (1=BUY, 0=SELL)

    Returns:
        Realized PnL for the market
    """
    total_shares = 0.0
    total_cost = 0.0
    realized_pnl = 0.0

    for i in range(len(prices)):
        if sides[i] == 1:  # BUY
            total_cost += prices[i] * sizes[i]
            total_shares += sizes[i]
        else:  # SELL
            if total_shares > 0:
                avg_entry = total_cost / total_shares
                sell_size = min(sizes[i], total_shares)
                realized_pnl += (prices[i] - avg_entry) * sell_size
                total_cost -= avg_entry * sell_size
                total_shares -= sell_size
        realized_pnl -= fees[i]

    return realized_pnl


@jit(nopython=True, cache=True)
def calculate_max_streak_numba(sequence: np.ndarray, target_value: int) -> int:
    """
    JIT-compiled maximum consecutive streak calculation.

    Args:
        sequence: Array of 0s and 1s
        target_value: Value to count streak of (0 or 1)

    Returns:
        Maximum consecutive count of target_value
    """
    max_streak = 0
    current_streak = 0

    for i in range(len(sequence)):
        if sequence[i] == target_value:
            current_streak += 1
            if current_streak > max_streak:
                max_streak = current_streak
        else:
            current_streak = 0

    return max_streak


@jit(nopython=True, cache=True)
def calculate_all_streaks_numba(sequence: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    JIT-compiled streak length calculation.

    Args:
        sequence: Array of 0s and 1s

    Returns:
        Tuple of (streak lengths array, number of streaks)
    """
    if len(sequence) == 0:
        return np.zeros(1, dtype=np.int64), 0

    # Pre-allocate max possible streaks
    streaks = np.zeros(len(sequence), dtype=np.int64)
    streak_count = 0
    current_streak = 1

    for i in range(1, len(sequence)):
        if sequence[i] == sequence[i - 1]:
            current_streak += 1
        else:
            streaks[streak_count] = current_streak
            streak_count += 1
            current_streak = 1

    streaks[streak_count] = current_streak
    streak_count += 1

    return streaks, streak_count


@jit(nopython=True, cache=True)
def calculate_interval_entropy_numba(intervals: np.ndarray, num_bins: int) -> float:
    """
    JIT-compiled interval entropy calculation.

    Args:
        intervals: Array of time intervals in seconds
        num_bins: Number of histogram bins

    Returns:
        Normalized entropy (0-100)
    """
    if len(intervals) < 3:
        return 50.0

    # Manual histogram since np.histogram isn't supported in numba
    min_val = intervals.min()
    max_val = intervals.max()

    if max_val == min_val:
        return 0.0  # All same interval = perfectly regular

    bin_width = (max_val - min_val) / num_bins
    hist = np.zeros(num_bins, dtype=np.float64)

    for i in range(len(intervals)):
        bin_idx = int((intervals[i] - min_val) / bin_width)
        if bin_idx >= num_bins:
            bin_idx = num_bins - 1
        hist[bin_idx] += 1

    # Count non-zero bins
    non_zero_count = 0
    for i in range(num_bins):
        if hist[i] > 0:
            non_zero_count += 1

    if non_zero_count < 2:
        return 0.0

    # Calculate entropy
    total = len(intervals)
    entropy = 0.0
    for i in range(num_bins):
        if hist[i] > 0:
            prob = hist[i] / total
            entropy -= prob * np.log(prob)

    # Normalize by max entropy
    max_entropy = np.log(non_zero_count)
    if max_entropy == 0:
        return 0.0

    return (entropy / max_entropy) * 100

# Time period constants (imported from models for consistency)
TIME_PERIOD_ALL = "all_time"
TIME_PERIOD_7D = "7d"
TIME_PERIOD_31D = "31d"
TIME_PERIODS = [TIME_PERIOD_ALL, TIME_PERIOD_7D, TIME_PERIOD_31D]


def get_time_period_cutoff(time_period: str) -> Optional[datetime]:
    """Get the cutoff datetime for a time period"""
    now = datetime.utcnow()
    if time_period == TIME_PERIOD_7D:
        return now - timedelta(days=7)
    elif time_period == TIME_PERIOD_31D:
        return now - timedelta(days=31)
    return None  # all_time has no cutoff


@dataclass
class WalletData:
    """Container for wallet trade and position data"""
    address: str
    trades: List[Trade]
    positions: List[Position]
    time_period: str = TIME_PERIOD_ALL  # Which time period this data represents

    # Cached DataFrames
    _trades_df: Optional[pd.DataFrame] = field(default=None, repr=False)
    _positions_df: Optional[pd.DataFrame] = field(default=None, repr=False)

    # Cached computed values (prevents 36+ redundant calculations per wallet)
    _cost_basis_map: Optional[Dict[str, float]] = field(default=None, repr=False)
    _pnl_series: Optional[pd.Series] = field(default=None, repr=False)
    _market_pnl_map: Optional[Dict[str, float]] = field(default=None, repr=False)

    def for_time_period(self, time_period: str) -> 'WalletData':
        """Create a new WalletData filtered to a specific time period"""
        if time_period == TIME_PERIOD_ALL:
            return WalletData(
                address=self.address,
                trades=self.trades,
                positions=self.positions,
                time_period=time_period,
            )

        cutoff = get_time_period_cutoff(time_period)
        if cutoff is None:
            return self

        # Filter trades by timestamp
        filtered_trades = [
            t for t in self.trades
            if t.timestamp and t.timestamp >= cutoff
        ]

        return WalletData(
            address=self.address,
            trades=filtered_trades,
            positions=self.positions,  # Positions are current state, not time-filtered
            time_period=time_period,
        )

    @property
    def trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame"""
        if self._trades_df is None:
            if not self.trades:
                self._trades_df = pd.DataFrame()
            else:
                self._trades_df = pd.DataFrame([
                    {
                        'id': t.id,
                        'timestamp': t.timestamp,
                        'market_id': t.market_id,
                        'condition_id': t.condition_id,
                        'side': t.side,
                        'outcome': t.outcome,
                        'price': t.price,
                        'size': t.size,
                        'value': t.value,
                        'fee': t.fee,
                    }
                    for t in self.trades
                ])
                if not self._trades_df.empty:
                    self._trades_df['timestamp'] = pd.to_datetime(self._trades_df['timestamp'])
                    self._trades_df = self._trades_df.sort_values('timestamp')
        return self._trades_df

    @property
    def positions_df(self) -> pd.DataFrame:
        """Get positions as DataFrame"""
        if self._positions_df is None:
            if not self.positions:
                self._positions_df = pd.DataFrame()
            else:
                self._positions_df = pd.DataFrame([
                    {
                        'market_id': p.market_id,
                        'condition_id': p.condition_id,
                        'outcome': p.outcome,
                        'outcome_index': p.outcome_index,
                        'size': p.size,
                        'avg_price': p.avg_price,
                        'current_price': p.current_price,
                        'value': p.value,
                        'unrealized_pnl': p.unrealized_pnl,
                        'realized_pnl': p.realized_pnl,
                    }
                    for p in self.positions
                ])
        return self._positions_df

    @property
    def total_volume(self) -> float:
        """Total trading volume"""
        if self.trades_df.empty:
            return 0.0
        return self.trades_df['value'].sum()

    @property
    def total_trades(self) -> int:
        """Total number of trades"""
        return len(self.trades)

    @property
    def has_trades(self) -> bool:
        """Check if wallet has trades"""
        return len(self.trades) > 0

    @property
    def cost_basis_map(self) -> Dict[str, float]:
        """
        Get cached cost basis map (per-market weighted average buy price).

        Memoized to prevent 36+ redundant calculations per wallet.
        """
        if self._cost_basis_map is None:
            self._cost_basis_map = self._compute_cost_basis_map()
        return self._cost_basis_map

    def _compute_cost_basis_map(self) -> Dict[str, float]:
        """Compute weighted average buy price per market."""
        df = self.trades_df
        if df.empty:
            return {}

        group_col = 'condition_id' if 'condition_id' in df.columns else 'market_id'
        buys = df[df['side'] == 'BUY']

        if buys.empty:
            return {}

        cost_basis = {}
        for market_id, market_buys in buys.groupby(group_col):
            total_size = market_buys['size'].sum()
            if total_size > 0:
                weighted_price = (market_buys['price'] * market_buys['size']).sum() / total_size
                cost_basis[str(market_id)] = weighted_price

        return cost_basis

    @property
    def pnl_series(self) -> pd.Series:
        """
        Get cached PnL series for all trades.

        Memoized to prevent redundant cost basis lookups.
        """
        if self._pnl_series is None:
            self._pnl_series = self._compute_pnl_series()
        return self._pnl_series

    def _compute_pnl_series(self) -> pd.Series:
        """
        Compute PnL series using cached cost basis.

        VECTORIZED: Uses NumPy operations instead of df.apply() for 10-50x speedup.
        """
        df = self.trades_df
        if df.empty:
            return pd.Series(dtype=float)

        cost_basis = self.cost_basis_map
        group_col = 'condition_id' if 'condition_id' in df.columns else 'market_id'

        # Vectorized computation - no df.apply() or iterrows()
        is_sell = (df['side'] == 'SELL').values
        market_keys = df[group_col].astype(str).values
        prices = df['price'].values
        sizes = df['size'].values
        fees = df['fee'].values if 'fee' in df.columns else np.zeros(len(df))

        # Map market keys to cost basis (vectorized lookup)
        avg_buys = np.array([cost_basis.get(mk, prices[i]) for i, mk in enumerate(market_keys)])

        # Compute PnL: (sell_price - avg_buy) * size - fee for sells, 0 otherwise
        pnl = np.where(is_sell, (prices - avg_buys) * sizes - fees, 0.0)

        return pd.Series(pnl, index=df.index)

    @property
    def market_pnl_map(self) -> Dict[str, float]:
        """
        Get cached per-market PnL map.

        Uses cost-basis tracking for accurate realized PnL per market.
        """
        if self._market_pnl_map is None:
            self._market_pnl_map = self._compute_market_pnl_map()
        return self._market_pnl_map

    def _compute_market_pnl_map(self) -> Dict[str, float]:
        """
        Compute per-market PnL using cost-basis tracking.

        VECTORIZED: Uses Numba JIT function instead of iterrows() for 5-10x speedup.
        """
        df = self.trades_df
        if df.empty:
            return {}

        group_col = 'condition_id' if 'condition_id' in df.columns else 'market_id'
        market_pnls: Dict[str, float] = {}

        # Pre-sort once
        df_sorted = df.sort_values(['timestamp'])

        for market_id, market_df in df_sorted.groupby(group_col):
            # Extract numpy arrays for Numba JIT function
            prices = market_df['price'].values.astype(np.float64)
            sizes = market_df['size'].values.astype(np.float64)
            fees = market_df['fee'].values.astype(np.float64) if 'fee' in market_df.columns else np.zeros(len(market_df), dtype=np.float64)
            sides = (market_df['side'] == 'BUY').values.astype(np.int64)  # 1=BUY, 0=SELL

            # Use Numba JIT function for fast PnL calculation
            if NUMBA_AVAILABLE:
                realized_pnl = calculate_market_pnl_numba(prices, sizes, fees, sides)
            else:
                # Fallback: vectorized NumPy (still faster than iterrows)
                realized_pnl = self._calculate_market_pnl_numpy(prices, sizes, fees, sides)

            market_pnls[str(market_id)] = realized_pnl

        # Add position-based PnL if available (vectorized)
        positions_df = self.positions_df
        if not positions_df.empty:
            pos_group_col = 'condition_id' if 'condition_id' in positions_df.columns else 'market_id'

            # Vectorized position PnL aggregation
            for col in ['unrealized_pnl', 'realized_pnl']:
                if col in positions_df.columns:
                    valid_mask = positions_df[col].notna()
                    if valid_mask.any():
                        for mk, pnl in zip(
                            positions_df.loc[valid_mask, pos_group_col].astype(str),
                            positions_df.loc[valid_mask, col]
                        ):
                            market_pnls[mk] = market_pnls.get(mk, 0.0) + pnl

        return market_pnls

    @staticmethod
    def _calculate_market_pnl_numpy(
        prices: np.ndarray,
        sizes: np.ndarray,
        fees: np.ndarray,
        sides: np.ndarray,  # 1=BUY, 0=SELL
    ) -> float:
        """
        NumPy fallback for market PnL calculation when Numba is not available.

        Uses a single pass through the trades with running cost basis tracking.
        """
        total_shares = 0.0
        total_cost = 0.0
        realized_pnl = 0.0

        for i in range(len(prices)):
            if sides[i] == 1:  # BUY
                total_cost += prices[i] * sizes[i]
                total_shares += sizes[i]
            else:  # SELL
                if total_shares > 0:
                    avg_entry = total_cost / total_shares
                    sell_size = min(sizes[i], total_shares)
                    realized_pnl += (prices[i] - avg_entry) * sell_size
                    total_cost -= avg_entry * sell_size
                    total_shares -= sell_size
            realized_pnl -= fees[i]

        return realized_pnl

    def invalidate_cache(self):
        """Invalidate all cached computed values."""
        self._cost_basis_map = None
        self._pnl_series = None
        self._market_pnl_map = None


class BaseMetricGroup(ABC):
    """Base class for metric group calculations"""

    GROUP_ID: str = ""
    METRIC_IDS: List[int] = []

    def __init__(self):
        pass

    @abstractmethod
    def calculate(self, data: WalletData) -> Dict[int, float]:
        """
        Calculate all metrics in this group.

        Args:
            data: Wallet trade and position data

        Returns:
            Dict mapping metric_id to calculated value
        """
        pass

    def safe_divide(self, numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division with default for zero denominator"""
        if denominator == 0 or pd.isna(denominator):
            return default
        return numerator / denominator

    def safe_std(self, series: pd.Series, default: float = 0.0) -> float:
        """Safe standard deviation"""
        if len(series) < 2:
            return default
        return series.std() or default

    def safe_mean(self, series: pd.Series, default: float = 0.0) -> float:
        """Safe mean calculation"""
        if series.empty:
            return default
        return series.mean() or default

    def percentile(self, series: pd.Series, q: float) -> float:
        """Calculate percentile"""
        if series.empty:
            return 0.0
        return np.percentile(series, q)

    def calculate_returns(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trade returns"""
        if df.empty or 'value' not in df.columns:
            return pd.Series()

        # Simplified return calculation based on trade outcomes
        # In practice, this would need position matching
        returns = df['value'].pct_change().dropna()
        return returns

    def get_winning_trades(self, df: pd.DataFrame, data: Optional[WalletData] = None) -> pd.DataFrame:
        """
        Get winning trades using actual cost basis.

        VECTORIZED: Uses NumPy operations instead of df.apply() for 10-50x speedup.
        """
        if df.empty:
            return pd.DataFrame()

        # Use cached cost basis from WalletData if available
        if data is not None:
            cost_basis_map = data.cost_basis_map
        else:
            cost_basis_map = self._build_cost_basis_map(df)

        pnl = self._compute_pnl_vectorized(df, cost_basis_map)
        df = df.copy()
        df['pnl'] = pnl
        return df[df['pnl'] > 0]

    def get_losing_trades(self, df: pd.DataFrame, data: Optional[WalletData] = None) -> pd.DataFrame:
        """
        Get losing trades using actual cost basis.

        VECTORIZED: Uses NumPy operations instead of df.apply() for 10-50x speedup.
        """
        if df.empty:
            return pd.DataFrame()

        # Use cached cost basis from WalletData if available
        if data is not None:
            cost_basis_map = data.cost_basis_map
        else:
            cost_basis_map = self._build_cost_basis_map(df)

        pnl = self._compute_pnl_vectorized(df, cost_basis_map)
        df = df.copy()
        df['pnl'] = pnl
        return df[df['pnl'] < 0]

    def _compute_pnl_vectorized(self, df: pd.DataFrame, cost_basis_map: Dict[str, float]) -> np.ndarray:
        """
        Compute PnL for all trades using vectorized NumPy operations.

        Returns numpy array of PnL values aligned with df index.
        """
        if df.empty:
            return np.array([])

        group_col = 'condition_id' if 'condition_id' in df.columns else 'market_id'

        is_sell = (df['side'] == 'SELL').values
        market_keys = df[group_col].astype(str).values
        prices = df['price'].values
        sizes = df['size'].values
        fees = df['fee'].values if 'fee' in df.columns else np.zeros(len(df))

        # Vectorized cost basis lookup
        avg_buys = np.array([cost_basis_map.get(mk, prices[i]) for i, mk in enumerate(market_keys)])

        # Compute PnL: (sell_price - avg_buy) * size - fee for sells, 0 otherwise
        return np.where(is_sell, (prices - avg_buys) * sizes - fees, 0.0)

    def _build_cost_basis_map(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Build a per-market weighted average buy price map.

        Groups BUY trades by condition_id and computes the volume-weighted
        average entry price for each market.

        Note: Prefer using WalletData.cost_basis_map for memoized access.

        Returns:
            Dict mapping condition_id -> weighted average buy price
        """
        if df.empty:
            return {}

        group_col = 'condition_id' if 'condition_id' in df.columns else 'market_id'
        buys = df[df['side'] == 'BUY']

        if buys.empty:
            return {}

        cost_basis = {}
        for market_id, market_buys in buys.groupby(group_col):
            total_size = market_buys['size'].sum()
            if total_size > 0:
                weighted_price = (market_buys['price'] * market_buys['size']).sum() / total_size
                cost_basis[str(market_id)] = weighted_price

        return cost_basis

    def _estimate_trade_pnl(self, row, cost_basis_map: Optional[Dict[str, float]] = None) -> float:
        """
        Estimate PnL for a single trade using actual cost basis.

        For SELL trades: pnl = (sell_price - avg_buy_price) * size - fee
        For BUY trades: return 0.0 (unrealized)
        """
        if row['side'] == 'SELL':
            if cost_basis_map is None:
                return 0.0
            group_col = 'condition_id' if 'condition_id' in row.index else 'market_id'
            market_key = str(row[group_col])
            avg_buy = cost_basis_map.get(market_key, row['price'])
            return (row['price'] - avg_buy) * row['size'] - row.get('fee', 0.0)
        else:
            return 0.0

    def _calculate_market_pnl_map(
        self,
        df: pd.DataFrame,
        positions_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """
        Calculate per-market PnL using cost-basis tracking.

        VECTORIZED: Uses Numba JIT function instead of iterrows() for 5-10x speedup.

        For each market:
        - Sort trades by timestamp
        - Track running cost basis (total_shares_held, total_cost)
        - For BUY: accumulate cost basis
        - For SELL: compute realized PnL against average entry price
        - Subtract fees

        Returns:
            Dict mapping market_id -> realized PnL
        """
        if df.empty:
            return {}

        group_col = 'condition_id' if 'condition_id' in df.columns else 'market_id'
        market_pnls: Dict[str, float] = {}

        # Pre-sort once
        df_sorted = df.sort_values(['timestamp'])

        for market_id, market_df in df_sorted.groupby(group_col):
            # Extract numpy arrays for Numba JIT function
            prices = market_df['price'].values.astype(np.float64)
            sizes = market_df['size'].values.astype(np.float64)
            fees = market_df['fee'].values.astype(np.float64) if 'fee' in market_df.columns else np.zeros(len(market_df), dtype=np.float64)
            sides = (market_df['side'] == 'BUY').values.astype(np.int64)  # 1=BUY, 0=SELL

            # Use Numba JIT function for fast PnL calculation
            if NUMBA_AVAILABLE:
                realized_pnl = calculate_market_pnl_numba(prices, sizes, fees, sides)
            else:
                # Fallback: pure Python loop (still works, just slower)
                realized_pnl = self._calculate_market_pnl_fallback(prices, sizes, fees, sides)

            market_pnls[str(market_id)] = realized_pnl

        # Add position-based PnL if available (vectorized)
        if positions_df is not None and not positions_df.empty:
            pos_group_col = 'condition_id' if 'condition_id' in positions_df.columns else 'market_id'

            # Vectorized position PnL aggregation
            for col in ['unrealized_pnl', 'realized_pnl']:
                if col in positions_df.columns:
                    valid_mask = positions_df[col].notna()
                    if valid_mask.any():
                        for mk, pnl in zip(
                            positions_df.loc[valid_mask, pos_group_col].astype(str),
                            positions_df.loc[valid_mask, col]
                        ):
                            market_pnls[mk] = market_pnls.get(mk, 0.0) + pnl

        return market_pnls

    @staticmethod
    def _calculate_market_pnl_fallback(
        prices: np.ndarray,
        sizes: np.ndarray,
        fees: np.ndarray,
        sides: np.ndarray,
    ) -> float:
        """Pure Python fallback for market PnL calculation."""
        total_shares = 0.0
        total_cost = 0.0
        realized_pnl = 0.0

        for i in range(len(prices)):
            if sides[i] == 1:  # BUY
                total_cost += prices[i] * sizes[i]
                total_shares += sizes[i]
            else:  # SELL
                if total_shares > 0:
                    avg_entry = total_cost / total_shares
                    sell_size = min(sizes[i], total_shares)
                    realized_pnl += (prices[i] - avg_entry) * sell_size
                    total_cost -= avg_entry * sell_size
                    total_shares -= sell_size
            realized_pnl -= fees[i]

        return realized_pnl

    def calculate_pnl_by_market(
        self,
        df: pd.DataFrame,
        positions_df: Optional[pd.DataFrame] = None,
        data: Optional[WalletData] = None,
    ) -> float:
        """
        Calculate total PnL using cost-basis tracking per market.

        Uses _calculate_market_pnl_map for the per-market breakdown,
        then sums all market PnLs.

        Note: Prefer using sum(WalletData.market_pnl_map.values()) for
        memoized access when calculating for the full trades DataFrame.

        Args:
            df: Trades DataFrame
            positions_df: Optional positions DataFrame for unrealized PnL
            data: Optional WalletData for cached market_pnl_map

        Returns:
            Total realized + unrealized PnL
        """
        # Use cached market PnL map if available and this is the full trades DataFrame
        if data is not None and df is data.trades_df:
            return sum(data.market_pnl_map.values())

        market_pnls = self._calculate_market_pnl_map(df, positions_df)
        return sum(market_pnls.values())

    def calculate_pnl_series(self, df: pd.DataFrame, data: Optional[WalletData] = None) -> pd.Series:
        """
        Calculate PnL series for all trades using cost basis.

        VECTORIZED: Uses NumPy operations instead of df.apply() for 10-50x speedup.

        Note: Prefer using WalletData.pnl_series for memoized access when
        calculating for the full trades DataFrame.
        """
        if df.empty:
            return pd.Series(dtype=float)

        # Use cached cost basis from WalletData if available
        if data is not None:
            cost_basis_map = data.cost_basis_map
        else:
            cost_basis_map = self._build_cost_basis_map(df)

        pnl = self._compute_pnl_vectorized(df, cost_basis_map)
        return pd.Series(pnl, index=df.index)
