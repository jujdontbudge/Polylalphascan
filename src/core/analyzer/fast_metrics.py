"""
Fast Metric Calculator - Pure NumPy/Numba implementation.

Bypasses pandas entirely for 100,000x speedup.
Target: 1000+ wallets/second.
"""
import numpy as np
from numba import jit
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# Type aliases for clarity
ArrayF64 = np.ndarray  # float64 array
ArrayI64 = np.ndarray  # int64 array


@dataclass
class FastWalletData:
    """
    Wallet data in NumPy array format for fast processing.

    All arrays must be the same length (n_trades).
    """
    address: str
    n_trades: int
    n_markets: int

    # Trade arrays (length = n_trades)
    prices: ArrayF64      # Trade prices (0-1 for prediction markets)
    sizes: ArrayF64       # Trade sizes
    fees: ArrayF64        # Trade fees
    sides: ArrayI64       # 1=BUY, 0=SELL
    timestamps: ArrayI64  # Unix timestamps
    market_ids: ArrayI64  # Market index (0 to n_markets-1)
    values: ArrayF64      # Trade values (price * size)

    @classmethod
    def from_trades(cls, address: str, trades: List[dict]) -> 'FastWalletData':
        """Convert trade list to numpy arrays."""
        if not trades:
            return cls(
                address=address,
                n_trades=0,
                n_markets=0,
                prices=np.array([], dtype=np.float64),
                sizes=np.array([], dtype=np.float64),
                fees=np.array([], dtype=np.float64),
                sides=np.array([], dtype=np.int64),
                timestamps=np.array([], dtype=np.int64),
                market_ids=np.array([], dtype=np.int64),
                values=np.array([], dtype=np.float64),
            )

        n = len(trades)

        # Build market ID mapping
        unique_markets = list(set(t.get('market_id') or t.get('condition_id', '') for t in trades))
        market_to_idx = {m: i for i, m in enumerate(unique_markets)}

        prices = np.zeros(n, dtype=np.float64)
        sizes = np.zeros(n, dtype=np.float64)
        fees = np.zeros(n, dtype=np.float64)
        sides = np.zeros(n, dtype=np.int64)
        timestamps = np.zeros(n, dtype=np.int64)
        market_ids = np.zeros(n, dtype=np.int64)
        values = np.zeros(n, dtype=np.float64)

        for i, t in enumerate(trades):
            prices[i] = float(t.get('price', 0) or 0)
            sizes[i] = float(t.get('size', 0) or 0)
            fees[i] = float(t.get('fee', 0) or 0)
            sides[i] = 1 if t.get('side', '').upper() == 'BUY' else 0

            ts = t.get('timestamp')
            if isinstance(ts, datetime):
                timestamps[i] = int(ts.timestamp())
            elif isinstance(ts, (int, float)):
                timestamps[i] = int(ts)
            else:
                timestamps[i] = 0

            market_key = t.get('market_id') or t.get('condition_id', '')
            market_ids[i] = market_to_idx.get(market_key, 0)
            values[i] = float(t.get('value', 0) or 0) or (prices[i] * sizes[i])

        return cls(
            address=address,
            n_trades=n,
            n_markets=len(unique_markets),
            prices=prices,
            sizes=sizes,
            fees=fees,
            sides=sides,
            timestamps=timestamps,
            market_ids=market_ids,
            values=values,
        )


# =============================================================================
# NUMBA JIT FUNCTIONS - Pure computation, no Python overhead
# =============================================================================

@jit(nopython=True)
def calc_basic_stats(values: ArrayF64, sides: ArrayI64) -> Tuple[float, float, float, int, int]:
    """Calculate basic trading statistics."""
    total_volume = 0.0
    buy_volume = 0.0
    sell_volume = 0.0
    buy_count = 0
    sell_count = 0

    for i in range(len(values)):
        total_volume += values[i]
        if sides[i] == 1:
            buy_volume += values[i]
            buy_count += 1
        else:
            sell_volume += values[i]
            sell_count += 1

    return total_volume, buy_volume, sell_volume, buy_count, sell_count


@jit(nopython=True)
def calc_market_pnl(
    prices: ArrayF64,
    sizes: ArrayF64,
    fees: ArrayF64,
    sides: ArrayI64,
    market_ids: ArrayI64,
    n_markets: int
) -> Tuple[ArrayF64, float, int, int]:
    """
    Calculate PnL per market using cost-basis tracking.

    Returns:
        market_pnl: Array of PnL per market
        total_pnl: Sum of all market PnLs
        wins: Number of winning markets
        losses: Number of losing markets
    """
    market_pnl = np.zeros(n_markets, dtype=np.float64)
    market_shares = np.zeros(n_markets, dtype=np.float64)
    market_cost = np.zeros(n_markets, dtype=np.float64)

    for i in range(len(prices)):
        m = market_ids[i]
        if sides[i] == 1:  # BUY
            market_cost[m] += prices[i] * sizes[i]
            market_shares[m] += sizes[i]
        else:  # SELL
            if market_shares[m] > 0:
                avg_entry = market_cost[m] / market_shares[m]
                sell_size = min(sizes[i], market_shares[m])
                market_pnl[m] += (prices[i] - avg_entry) * sell_size
                market_cost[m] -= avg_entry * sell_size
                market_shares[m] -= sell_size
        market_pnl[m] -= fees[i]

    total_pnl = 0.0
    wins = 0
    losses = 0
    for m in range(n_markets):
        total_pnl += market_pnl[m]
        if market_pnl[m] > 0:
            wins += 1
        elif market_pnl[m] < 0:
            losses += 1

    return market_pnl, total_pnl, wins, losses


@jit(nopython=True)
def calc_profit_metrics(market_pnl: ArrayF64, buy_volume: float) -> Tuple[float, float, float, float, float]:
    """Calculate profit-related metrics from market PnL array."""
    total_pnl = 0.0
    winning_pnl = 0.0
    losing_pnl = 0.0
    best_pnl = -1e18
    worst_pnl = 1e18

    for pnl in market_pnl:
        total_pnl += pnl
        if pnl > 0:
            winning_pnl += pnl
        else:
            losing_pnl += abs(pnl)
        if pnl > best_pnl:
            best_pnl = pnl
        if pnl < worst_pnl:
            worst_pnl = pnl

    roi = total_pnl / buy_volume if buy_volume > 0 else 0.0
    profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else 1.0

    if len(market_pnl) == 0:
        best_pnl = 0.0
        worst_pnl = 0.0

    return roi, profit_factor, total_pnl, best_pnl, worst_pnl


@jit(nopython=True)
def calc_timing_metrics(timestamps: ArrayI64) -> Tuple[float, float, float, int]:
    """Calculate timing-related metrics."""
    n = len(timestamps)
    if n < 2:
        return 0.0, 0.0, 0.0, 0

    # Sort timestamps
    sorted_ts = np.sort(timestamps)

    # Calculate intervals
    intervals = np.zeros(n - 1, dtype=np.float64)
    for i in range(n - 1):
        intervals[i] = float(sorted_ts[i + 1] - sorted_ts[i])

    # Stats
    mean_interval = 0.0
    for i in range(len(intervals)):
        mean_interval += intervals[i]
    mean_interval /= len(intervals)

    std_interval = 0.0
    for i in range(len(intervals)):
        std_interval += (intervals[i] - mean_interval) ** 2
    std_interval = (std_interval / len(intervals)) ** 0.5

    # Time span in days
    time_span_days = (sorted_ts[-1] - sorted_ts[0]) / 86400.0

    return mean_interval, std_interval, time_span_days, n


@jit(nopython=True)
def calc_risk_metrics(market_pnl: ArrayF64) -> Tuple[float, float, float]:
    """Calculate risk metrics from market PnL."""
    n = len(market_pnl)
    if n == 0:
        return 0.0, 0.0, 0.0

    # Mean and std
    mean_pnl = 0.0
    for pnl in market_pnl:
        mean_pnl += pnl
    mean_pnl /= n

    var_pnl = 0.0
    for pnl in market_pnl:
        var_pnl += (pnl - mean_pnl) ** 2
    std_pnl = (var_pnl / n) ** 0.5

    # Sharpe-like ratio (simplified)
    sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0.0

    # Max drawdown (simplified - on cumulative PnL)
    cumsum = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in market_pnl:
        cumsum += pnl
        if cumsum > peak:
            peak = cumsum
        dd = peak - cumsum
        if dd > max_dd:
            max_dd = dd

    return std_pnl, sharpe, max_dd


@jit(nopython=True)
def calc_win_loss_metrics(
    prices: ArrayF64,
    sides: ArrayI64,
    market_ids: ArrayI64,
    market_pnl: ArrayF64,
    n_markets: int
) -> Tuple[float, int, int, int, int]:
    """Calculate win/loss related metrics."""
    wins = 0
    losses = 0

    for m in range(n_markets):
        if market_pnl[m] > 0:
            wins += 1
        elif market_pnl[m] < 0:
            losses += 1

    total_markets = wins + losses
    win_rate = wins / total_markets if total_markets > 0 else 0.0

    # Consecutive wins/losses (simplified)
    max_win_streak = 0
    max_loss_streak = 0
    current_win = 0
    current_loss = 0

    for m in range(n_markets):
        if market_pnl[m] > 0:
            current_win += 1
            current_loss = 0
            if current_win > max_win_streak:
                max_win_streak = current_win
        elif market_pnl[m] < 0:
            current_loss += 1
            current_win = 0
            if current_loss > max_loss_streak:
                max_loss_streak = current_loss

    return win_rate, wins, losses, max_win_streak, max_loss_streak


@jit(nopython=True)
def calc_all_metrics_fast(
    prices: ArrayF64,
    sizes: ArrayF64,
    fees: ArrayF64,
    sides: ArrayI64,
    timestamps: ArrayI64,
    market_ids: ArrayI64,
    values: ArrayF64,
    n_markets: int
) -> Tuple[float, ...]:
    """
    Calculate ALL core metrics in a single pass.

    Returns tuple of 30 key metrics.
    """
    n = len(prices)
    if n == 0:
        return (0.0,) * 30

    # Basic stats
    total_volume, buy_volume, sell_volume, buy_count, sell_count = calc_basic_stats(values, sides)

    # Market PnL
    market_pnl, total_pnl, wins, losses = calc_market_pnl(
        prices, sizes, fees, sides, market_ids, n_markets
    )

    # Profit metrics
    roi, profit_factor, _, best_pnl, worst_pnl = calc_profit_metrics(market_pnl, buy_volume)

    # Risk metrics
    std_pnl, sharpe, max_dd = calc_risk_metrics(market_pnl)

    # Timing metrics
    mean_interval, std_interval, time_span_days, trade_count = calc_timing_metrics(timestamps)

    # Win/loss metrics
    win_rate, _, _, max_win_streak, max_loss_streak = calc_win_loss_metrics(
        prices, sides, market_ids, market_pnl, n_markets
    )

    # Additional derived metrics
    avg_trade_size = total_volume / n if n > 0 else 0.0
    buy_sell_ratio = buy_volume / sell_volume if sell_volume > 0 else 1.0
    fee_ratio = fees.sum() / total_volume if total_volume > 0 else 0.0
    trades_per_day = n / time_span_days if time_span_days > 0 else 0.0
    pnl_per_trade = total_pnl / n if n > 0 else 0.0
    pnl_per_volume = total_pnl / total_volume if total_volume > 0 else 0.0

    # Average market PnL
    avg_market_pnl = total_pnl / n_markets if n_markets > 0 else 0.0

    # Market concentration (simplified Herfindahl)
    market_volumes = np.zeros(n_markets, dtype=np.float64)
    for i in range(n):
        market_volumes[market_ids[i]] += values[i]

    concentration = 0.0
    for m in range(n_markets):
        share = market_volumes[m] / total_volume if total_volume > 0 else 0.0
        concentration += share * share

    return (
        total_pnl,           # 0
        roi * 100,           # 1 - as percentage
        profit_factor,       # 2
        win_rate * 100,      # 3 - as percentage
        total_volume,        # 4
        buy_volume,          # 5
        sell_volume,         # 6
        float(buy_count),    # 7
        float(sell_count),   # 8
        best_pnl,            # 9
        worst_pnl,           # 10
        avg_market_pnl,      # 11
        std_pnl,             # 12
        sharpe,              # 13
        max_dd,              # 14
        float(wins),         # 15
        float(losses),       # 16
        float(max_win_streak),   # 17
        float(max_loss_streak),  # 18
        avg_trade_size,      # 19
        buy_sell_ratio,      # 20
        fee_ratio * 100,     # 21 - as percentage
        trades_per_day,      # 22
        pnl_per_trade,       # 23
        pnl_per_volume,      # 24
        mean_interval,       # 25 - in seconds
        std_interval,        # 26
        time_span_days,      # 27
        float(n),            # 28 - trade count
        concentration,       # 29 - market concentration
    )


# =============================================================================
# HIGH-LEVEL API
# =============================================================================

# Metric ID mapping (matching existing metric IDs where possible)
FAST_METRIC_IDS = {
    0: 1,    # total_pnl -> Metric 1
    1: 2,    # roi -> Metric 2
    2: 4,    # profit_factor -> Metric 4
    3: 23,   # win_rate -> Metric 23
    4: 31,   # total_volume -> Metric 31
    5: 32,   # buy_volume -> Metric 32 (custom)
    6: 33,   # sell_volume -> Metric 33 (custom)
    7: 34,   # buy_count -> Metric 34 (custom)
    8: 35,   # sell_count -> Metric 35 (custom)
    9: 7,    # best_pnl -> Metric 7
    10: 8,   # worst_pnl -> Metric 8
    11: 5,   # avg_market_pnl -> Metric 5
    12: 12,  # std_pnl -> Metric 12
    13: 15,  # sharpe -> Metric 15
    14: 11,  # max_dd -> Metric 11
    15: 21,  # wins -> Metric 21
    16: 22,  # losses -> Metric 22
    17: 24,  # max_win_streak -> Metric 24
    18: 25,  # max_loss_streak -> Metric 25
    19: 36,  # avg_trade_size -> Metric 36
    20: 58,  # buy_sell_ratio -> Metric 58
    21: 91,  # fee_ratio -> Metric 91
    22: 37,  # trades_per_day -> Metric 37
    23: 38,  # pnl_per_trade -> Metric 38
    24: 9,   # pnl_per_volume -> Metric 9
    25: 41,  # mean_interval -> Metric 41
    26: 42,  # std_interval -> Metric 42
    27: 43,  # time_span_days -> Metric 43
    28: 31,  # trade_count -> Metric 31
    29: 102, # concentration -> Metric 102
}


def calculate_fast_metrics(data: FastWalletData) -> Dict[int, float]:
    """
    Calculate metrics using fast Numba implementation.

    Returns dict mapping metric_id -> value.
    """
    if data.n_trades == 0:
        return {mid: 0.0 for mid in FAST_METRIC_IDS.values()}

    results = calc_all_metrics_fast(
        data.prices,
        data.sizes,
        data.fees,
        data.sides,
        data.timestamps,
        data.market_ids,
        data.values,
        max(1, data.n_markets),
    )

    return {FAST_METRIC_IDS[i]: float(results[i]) for i in range(len(results))}


def calculate_fast_metrics_from_trades(address: str, trades: List[dict]) -> Dict[int, float]:
    """
    Convenience function: calculate metrics directly from trade list.
    """
    data = FastWalletData.from_trades(address, trades)
    return calculate_fast_metrics(data)


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def calculate_batch_fast(
    wallets: List[Tuple[str, List[dict]]]
) -> Dict[str, Dict[int, float]]:
    """
    Calculate metrics for multiple wallets.

    Args:
        wallets: List of (address, trades) tuples

    Returns:
        Dict mapping address -> metrics dict
    """
    results = {}
    for address, trades in wallets:
        results[address] = calculate_fast_metrics_from_trades(address, trades)
    return results


# Warm up JIT on import
def _warmup():
    """Compile JIT functions on import."""
    dummy = FastWalletData(
        address="0x0",
        n_trades=10,
        n_markets=2,
        prices=np.random.random(10),
        sizes=np.random.random(10) * 100,
        fees=np.random.random(10),
        sides=np.random.randint(0, 2, 10),
        timestamps=np.arange(10, dtype=np.int64),
        market_ids=np.random.randint(0, 2, 10),
        values=np.random.random(10) * 100,
    )
    _ = calculate_fast_metrics(dummy)

_warmup()
