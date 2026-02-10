"""
Numba Intermediate Calculations for Ultra-Fast Metrics Engine.

This module provides shared intermediate calculations that are used by multiple
metrics. By computing these once and caching them, we avoid 80% of redundant work.

Target: 50,000+ wallets/second for all 117 metrics.

Computation Hierarchy:
    Level 1: Basic Stats (volume, counts)
        ↓
    Level 2: Per-Market Arrays (pnl, volume, counts)
        ↓
    Level 3: Win/Loss Stats (wins, losses, win_rate)
        ↓
    Level 4: Time-Based (intervals, hold times)
        ↓
    Level 5: Returns & Risk (returns, drawdowns, sharpe)
"""
import numpy as np
from numba import jit
from typing import Tuple

# Type aliases for clarity
ArrayF64 = np.ndarray  # float64 array
ArrayI64 = np.ndarray  # int64 array


# =============================================================================
# LEVEL 1: Basic Aggregations
# =============================================================================

@jit(nopython=True, cache=True)
def calc_basic_stats(
    values: ArrayF64,
    fees: ArrayF64,
    sides: ArrayI64,
) -> Tuple[float, float, float, int, int, float]:
    """
    Calculate basic volume and count statistics.

    Args:
        values: Trade values (price * size)
        fees: Trade fees
        sides: 1=BUY, 0=SELL

    Returns:
        Tuple of (total_volume, buy_volume, sell_volume, buy_count, sell_count, total_fees)
    """
    n = len(values)
    total_volume = 0.0
    buy_volume = 0.0
    sell_volume = 0.0
    buy_count = 0
    sell_count = 0
    total_fees = 0.0

    for i in range(n):
        total_volume += values[i]
        total_fees += fees[i]
        if sides[i] == 1:  # BUY
            buy_volume += values[i]
            buy_count += 1
        else:  # SELL
            sell_volume += values[i]
            sell_count += 1

    return total_volume, buy_volume, sell_volume, buy_count, sell_count, total_fees


# =============================================================================
# LEVEL 2: Per-Market Calculations
# =============================================================================

@jit(nopython=True, cache=True)
def calc_market_arrays(
    prices: ArrayF64,
    sizes: ArrayF64,
    fees: ArrayF64,
    sides: ArrayI64,
    market_ids: ArrayI64,
    n_markets: int,
) -> Tuple[ArrayF64, ArrayF64, ArrayI64, ArrayF64, ArrayF64]:
    """
    Calculate per-market PnL, volume, and counts using cost-basis tracking.

    Args:
        prices: Trade prices (0-1 for prediction markets)
        sizes: Trade sizes
        fees: Trade fees
        sides: 1=BUY, 0=SELL
        market_ids: Market index (0 to n_markets-1)
        n_markets: Number of unique markets

    Returns:
        Tuple of (market_pnl, market_volume, market_count, market_cost, market_shares)
    """
    market_pnl = np.zeros(n_markets, dtype=np.float64)
    market_volume = np.zeros(n_markets, dtype=np.float64)
    market_count = np.zeros(n_markets, dtype=np.int64)
    market_cost = np.zeros(n_markets, dtype=np.float64)
    market_shares = np.zeros(n_markets, dtype=np.float64)

    n = len(prices)
    for i in range(n):
        m = market_ids[i]
        value = prices[i] * sizes[i]
        market_volume[m] += value
        market_count[m] += 1

        # Cost basis tracking for accurate PnL
        if sides[i] == 1:  # BUY
            market_cost[m] += value
            market_shares[m] += sizes[i]
        else:  # SELL
            if market_shares[m] > 0:
                avg_entry = market_cost[m] / market_shares[m]
                sell_size = min(sizes[i], market_shares[m])
                market_pnl[m] += (prices[i] - avg_entry) * sell_size
                market_cost[m] -= avg_entry * sell_size
                market_shares[m] -= sell_size

        # Subtract fees from PnL
        market_pnl[m] -= fees[i]

    return market_pnl, market_volume, market_count, market_cost, market_shares


@jit(nopython=True, cache=True)
def calc_market_buy_volume(
    values: ArrayF64,
    sides: ArrayI64,
    market_ids: ArrayI64,
    n_markets: int,
) -> ArrayF64:
    """Calculate buy volume per market."""
    market_buy_volume = np.zeros(n_markets, dtype=np.float64)

    for i in range(len(values)):
        if sides[i] == 1:  # BUY
            market_buy_volume[market_ids[i]] += values[i]

    return market_buy_volume


# =============================================================================
# LEVEL 3: Win/Loss Statistics
# =============================================================================

@jit(nopython=True, cache=True)
def calc_win_loss_stats(market_pnl: ArrayF64) -> Tuple[float, int, int, float, float, float]:
    """
    Calculate win/loss statistics from market PnL array.

    Args:
        market_pnl: PnL per market

    Returns:
        Tuple of (total_pnl, wins, losses, win_rate, winning_pnl, losing_pnl)
    """
    n = len(market_pnl)
    total_pnl = 0.0
    wins = 0
    losses = 0
    winning_pnl = 0.0
    losing_pnl = 0.0

    for i in range(n):
        pnl = market_pnl[i]
        total_pnl += pnl
        if pnl > 0:
            wins += 1
            winning_pnl += pnl
        elif pnl < 0:
            losses += 1
            losing_pnl += abs(pnl)

    total_markets = wins + losses
    win_rate = wins / total_markets if total_markets > 0 else 0.0

    return total_pnl, wins, losses, win_rate, winning_pnl, losing_pnl


@jit(nopython=True, cache=True)
def calc_streak_stats(market_pnl: ArrayF64) -> Tuple[int, int, float]:
    """
    Calculate win/loss streak statistics.

    Args:
        market_pnl: PnL per market (in order of completion)

    Returns:
        Tuple of (max_win_streak, max_loss_streak, avg_streak_length)
    """
    n = len(market_pnl)
    if n == 0:
        return 0, 0, 0.0

    max_win_streak = 0
    max_loss_streak = 0
    current_win = 0
    current_loss = 0

    streak_lengths = np.zeros(n, dtype=np.int64)
    streak_count = 0
    current_streak = 1

    for i in range(n):
        # Win/loss streaks
        if market_pnl[i] > 0:
            current_win += 1
            current_loss = 0
            if current_win > max_win_streak:
                max_win_streak = current_win
        elif market_pnl[i] < 0:
            current_loss += 1
            current_win = 0
            if current_loss > max_loss_streak:
                max_loss_streak = current_loss
        else:
            current_win = 0
            current_loss = 0

        # All streaks for average
        if i > 0:
            prev_win = market_pnl[i-1] > 0
            curr_win = market_pnl[i] > 0
            if prev_win == curr_win and market_pnl[i] != 0 and market_pnl[i-1] != 0:
                current_streak += 1
            else:
                if current_streak > 0:
                    streak_lengths[streak_count] = current_streak
                    streak_count += 1
                current_streak = 1

    # Add final streak
    if current_streak > 0:
        streak_lengths[streak_count] = current_streak
        streak_count += 1

    # Calculate average streak length
    avg_streak = 0.0
    if streak_count > 0:
        total = 0
        for i in range(streak_count):
            total += streak_lengths[i]
        avg_streak = total / streak_count

    return max_win_streak, max_loss_streak, avg_streak


# =============================================================================
# LEVEL 4: Time-Based Calculations
# =============================================================================

@jit(nopython=True, cache=True)
def calc_time_stats(timestamps: ArrayI64) -> Tuple[ArrayF64, float, float, float, float]:
    """
    Calculate time interval statistics.

    Args:
        timestamps: Unix timestamps (int64)

    Returns:
        Tuple of (intervals, time_span_days, mean_interval, std_interval, median_interval)
    """
    n = len(timestamps)
    if n < 2:
        return np.zeros(1, dtype=np.float64), 0.0, 0.0, 0.0, 0.0

    # Sort timestamps
    sorted_ts = np.sort(timestamps)

    # Calculate intervals
    intervals = np.zeros(n - 1, dtype=np.float64)
    for i in range(n - 1):
        intervals[i] = float(sorted_ts[i + 1] - sorted_ts[i])

    # Time span in days
    time_span_days = float(sorted_ts[-1] - sorted_ts[0]) / 86400.0

    # Mean interval
    mean_interval = 0.0
    for i in range(len(intervals)):
        mean_interval += intervals[i]
    mean_interval /= len(intervals)

    # Std interval
    var_interval = 0.0
    for i in range(len(intervals)):
        var_interval += (intervals[i] - mean_interval) ** 2
    std_interval = (var_interval / len(intervals)) ** 0.5

    # Median interval (simple sort-based)
    sorted_intervals = np.sort(intervals)
    mid = len(sorted_intervals) // 2
    if len(sorted_intervals) % 2 == 0:
        median_interval = (sorted_intervals[mid - 1] + sorted_intervals[mid]) / 2
    else:
        median_interval = sorted_intervals[mid]

    return intervals, time_span_days, mean_interval, std_interval, median_interval


@jit(nopython=True, cache=True)
def calc_per_market_hold_times(
    timestamps: ArrayI64,
    sides: ArrayI64,
    market_ids: ArrayI64,
    n_markets: int,
) -> Tuple[ArrayF64, int]:
    """
    Calculate hold times per market (time between trades in same market).

    Returns:
        Tuple of (hold_times_array, count)
    """
    # Pre-allocate max possible hold times
    max_hold_times = len(timestamps)
    hold_times = np.zeros(max_hold_times, dtype=np.float64)
    hold_count = 0

    # Track last trade timestamp per market
    last_timestamp = np.full(n_markets, -1, dtype=np.int64)

    for i in range(len(timestamps)):
        m = market_ids[i]
        ts = timestamps[i]

        if last_timestamp[m] >= 0:
            # Calculate hold time in hours
            diff_seconds = ts - last_timestamp[m]
            hold_times[hold_count] = diff_seconds / 3600.0
            hold_count += 1

        last_timestamp[m] = ts

    return hold_times, hold_count


@jit(nopython=True, cache=True)
def calc_hold_time_stats(hold_times: ArrayF64, count: int) -> Tuple[float, float, float, float]:
    """
    Calculate hold time statistics.

    Returns:
        Tuple of (avg_hold_hours, median_hold_hours, min_hold_seconds, max_hold_days)
    """
    if count == 0:
        return 0.0, 0.0, 0.0, 0.0

    # Get valid hold times
    valid = hold_times[:count]

    # Average
    avg_hold = 0.0
    for i in range(count):
        avg_hold += valid[i]
    avg_hold /= count

    # Median
    sorted_holds = np.sort(valid)
    mid = count // 2
    if count % 2 == 0:
        median_hold = (sorted_holds[mid - 1] + sorted_holds[mid]) / 2
    else:
        median_hold = sorted_holds[mid]

    # Min (convert to seconds)
    min_hold_seconds = sorted_holds[0] * 3600

    # Max (convert to days)
    max_hold_days = sorted_holds[count - 1] / 24

    return avg_hold, median_hold, min_hold_seconds, max_hold_days


# =============================================================================
# LEVEL 5: Returns & Risk Calculations
# =============================================================================

@jit(nopython=True, cache=True)
def calc_per_trade_pnl(
    prices: ArrayF64,
    sizes: ArrayF64,
    fees: ArrayF64,
    sides: ArrayI64,
    market_ids: ArrayI64,
    n_markets: int,
) -> ArrayF64:
    """
    Calculate PnL for each trade using cost-basis tracking.

    For sell trades: pnl = (sell_price - avg_entry) * size - fee
    For buy trades: pnl = -fee (just the fee impact)
    """
    n = len(prices)
    trade_pnl = np.zeros(n, dtype=np.float64)

    # Track cost basis per market
    market_cost = np.zeros(n_markets, dtype=np.float64)
    market_shares = np.zeros(n_markets, dtype=np.float64)

    for i in range(n):
        m = market_ids[i]

        if sides[i] == 1:  # BUY
            market_cost[m] += prices[i] * sizes[i]
            market_shares[m] += sizes[i]
            trade_pnl[i] = -fees[i]  # Fee impact only
        else:  # SELL
            if market_shares[m] > 0:
                avg_entry = market_cost[m] / market_shares[m]
                sell_size = min(sizes[i], market_shares[m])
                trade_pnl[i] = (prices[i] - avg_entry) * sell_size - fees[i]
                market_cost[m] -= avg_entry * sell_size
                market_shares[m] -= sell_size
            else:
                trade_pnl[i] = -fees[i]

    return trade_pnl


@jit(nopython=True, cache=True)
def calc_returns_and_risk(
    trade_pnl: ArrayF64,
    values: ArrayF64,
) -> Tuple[ArrayF64, ArrayF64, float, float, float, float, float]:
    """
    Calculate returns series and risk metrics.

    Args:
        trade_pnl: PnL per trade
        values: Trade values (for return calculation)

    Returns:
        Tuple of (returns, cumulative_returns, max_drawdown, volatility, sharpe,
                  downside_vol, sortino)
    """
    n = len(trade_pnl)
    if n == 0:
        return (np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64),
                0.0, 0.0, 0.0, 0.0, 0.0)

    # Returns as PnL / value
    returns = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if values[i] > 0:
            returns[i] = trade_pnl[i] / values[i]

    # Cumulative returns (1 + r1) * (1 + r2) * ...
    cum_returns = np.zeros(n, dtype=np.float64)
    cum_returns[0] = 1.0 + returns[0]
    for i in range(1, n):
        cum_returns[i] = cum_returns[i-1] * (1.0 + returns[i])

    # Max drawdown
    peak = cum_returns[0]
    max_dd = 0.0
    for i in range(n):
        if cum_returns[i] > peak:
            peak = cum_returns[i]
        dd = (peak - cum_returns[i]) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    # Mean return
    mean_ret = 0.0
    for r in returns:
        mean_ret += r
    mean_ret /= n

    # Volatility (standard deviation)
    var_sum = 0.0
    for r in returns:
        var_sum += (r - mean_ret) ** 2
    volatility = (var_sum / n) ** 0.5 if n > 0 else 0.0

    # Sharpe ratio (simplified, 0 risk-free rate)
    sharpe = mean_ret / volatility if volatility > 0 else 0.0

    # Downside volatility (only negative returns)
    downside_sum = 0.0
    downside_count = 0
    for r in returns:
        if r < 0:
            downside_sum += r ** 2
            downside_count += 1

    downside_vol = (downside_sum / downside_count) ** 0.5 if downside_count > 0 else 0.0

    # Sortino ratio
    sortino = mean_ret / downside_vol if downside_vol > 0 else 0.0

    return returns, cum_returns, max_dd, volatility, sharpe, downside_vol, sortino


@jit(nopython=True, cache=True)
def calc_drawdown_stats(
    cum_returns: ArrayF64,
) -> Tuple[float, float, int]:
    """
    Calculate detailed drawdown statistics.

    Returns:
        Tuple of (max_drawdown, avg_drawdown, max_drawdown_duration)
    """
    n = len(cum_returns)
    if n == 0:
        return 0.0, 0.0, 0

    peak = cum_returns[0]
    max_dd = 0.0
    dd_sum = 0.0
    dd_count = 0

    current_dd_duration = 0
    max_dd_duration = 0
    in_drawdown = False

    for i in range(n):
        if cum_returns[i] > peak:
            peak = cum_returns[i]
            if in_drawdown:
                if current_dd_duration > max_dd_duration:
                    max_dd_duration = current_dd_duration
                current_dd_duration = 0
                in_drawdown = False

        if peak > 0:
            dd = (peak - cum_returns[i]) / peak
        else:
            dd = 0.0

        if dd > max_dd:
            max_dd = dd

        if dd > 0:
            dd_sum += dd
            dd_count += 1
            in_drawdown = True
            current_dd_duration += 1

    # Check final drawdown duration
    if current_dd_duration > max_dd_duration:
        max_dd_duration = current_dd_duration

    avg_dd = dd_sum / dd_count if dd_count > 0 else 0.0

    return max_dd, avg_dd, max_dd_duration


@jit(nopython=True, cache=True)
def calc_var_metrics(returns: ArrayF64, percentile: float = 5.0) -> Tuple[float, float]:
    """
    Calculate Value at Risk and Conditional VaR.

    Args:
        returns: Returns array
        percentile: VaR percentile (default 5%)

    Returns:
        Tuple of (var, cvar)
    """
    n = len(returns)
    if n < 10:
        return 0.0, 0.0

    # Sort returns
    sorted_returns = np.sort(returns)

    # VaR at percentile
    idx = int(n * percentile / 100)
    if idx >= n:
        idx = n - 1
    var = sorted_returns[idx]

    # Conditional VaR (expected shortfall) - average of returns below VaR
    cvar_sum = 0.0
    cvar_count = 0
    for i in range(idx + 1):
        cvar_sum += sorted_returns[i]
        cvar_count += 1

    cvar = cvar_sum / cvar_count if cvar_count > 0 else var

    return var, cvar


# =============================================================================
# AGGREGATED INTERMEDIATE CALCULATION
# =============================================================================

@jit(nopython=True, cache=True)
def compute_all_intermediates(
    prices: ArrayF64,
    sizes: ArrayF64,
    fees: ArrayF64,
    sides: ArrayI64,
    timestamps: ArrayI64,
    market_ids: ArrayI64,
    values: ArrayF64,
    n_markets: int,
) -> Tuple[
    # Level 1: Basic stats
    float, float, float, int, int, float,
    # Level 2: Per-market (as scalars for key metrics)
    float, float, float, float,  # total_pnl, best_pnl, worst_pnl, avg_market_pnl
    # Level 3: Win/loss
    int, int, float, float, float,  # wins, losses, win_rate, winning_pnl, losing_pnl
    int, int, float,  # max_win_streak, max_loss_streak, avg_streak
    # Level 4: Time
    float, float, float, float,  # time_span_days, mean_interval, std_interval, median_interval
    float, float, float, float,  # avg_hold, median_hold, min_hold_sec, max_hold_days
    # Level 5: Risk
    float, float, float, float, float,  # max_dd, volatility, sharpe, downside_vol, sortino
    float, float, int,  # avg_dd, max_dd_duration (actually returned as int), recovery_factor placeholder
    float, float,  # var_5, cvar_5
]:
    """
    Compute all intermediate values in one optimized pass.

    This is the main entry point that computes all levels of intermediates
    in the correct dependency order.
    """
    n = len(prices)

    # Default values for empty data
    if n == 0:
        return (
            # Level 1
            0.0, 0.0, 0.0, 0, 0, 0.0,
            # Level 2
            0.0, 0.0, 0.0, 0.0,
            # Level 3
            0, 0, 0.0, 0.0, 0.0,
            0, 0, 0.0,
            # Level 4
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            # Level 5
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0,
            0.0, 0.0,
        )

    # Level 1: Basic stats
    total_volume, buy_volume, sell_volume, buy_count, sell_count, total_fees = \
        calc_basic_stats(values, fees, sides)

    # Level 2: Per-market arrays
    market_pnl, market_volume, market_count, market_cost, market_shares = \
        calc_market_arrays(prices, sizes, fees, sides, market_ids, n_markets)

    # Level 2 aggregated metrics
    total_pnl = 0.0
    best_pnl = -1e18
    worst_pnl = 1e18
    for m in range(n_markets):
        total_pnl += market_pnl[m]
        if market_pnl[m] > best_pnl:
            best_pnl = market_pnl[m]
        if market_pnl[m] < worst_pnl:
            worst_pnl = market_pnl[m]

    if n_markets == 0:
        best_pnl = 0.0
        worst_pnl = 0.0

    avg_market_pnl = total_pnl / n_markets if n_markets > 0 else 0.0

    # Level 3: Win/loss stats
    _, wins, losses, win_rate, winning_pnl, losing_pnl = calc_win_loss_stats(market_pnl)
    max_win_streak, max_loss_streak, avg_streak = calc_streak_stats(market_pnl)

    # Level 4: Time stats
    intervals, time_span_days, mean_interval, std_interval, median_interval = \
        calc_time_stats(timestamps)

    hold_times, hold_count = calc_per_market_hold_times(
        timestamps, sides, market_ids, n_markets
    )
    avg_hold, median_hold, min_hold_sec, max_hold_days = calc_hold_time_stats(
        hold_times, hold_count
    )

    # Level 5: Returns and risk
    trade_pnl = calc_per_trade_pnl(prices, sizes, fees, sides, market_ids, n_markets)
    returns, cum_returns, max_dd, volatility, sharpe, downside_vol, sortino = \
        calc_returns_and_risk(trade_pnl, values)

    avg_dd, _, max_dd_duration = calc_drawdown_stats(cum_returns)
    var_5, cvar_5 = calc_var_metrics(returns, 5.0)

    # Recovery factor placeholder (will be calculated in metrics)
    recovery_placeholder = 0

    return (
        # Level 1
        total_volume, buy_volume, sell_volume, buy_count, sell_count, total_fees,
        # Level 2
        total_pnl, best_pnl, worst_pnl, avg_market_pnl,
        # Level 3
        wins, losses, win_rate, winning_pnl, losing_pnl,
        max_win_streak, max_loss_streak, avg_streak,
        # Level 4
        time_span_days, mean_interval, std_interval, median_interval,
        avg_hold, median_hold, min_hold_sec, max_hold_days,
        # Level 5
        max_dd, volatility, sharpe, downside_vol, sortino,
        avg_dd, max_dd_duration, recovery_placeholder,
        var_5, cvar_5,
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@jit(nopython=True, cache=True)
def calc_median(arr: ArrayF64) -> float:
    """Calculate median of an array."""
    n = len(arr)
    if n == 0:
        return 0.0

    sorted_arr = np.sort(arr)
    mid = n // 2

    if n % 2 == 0:
        return (sorted_arr[mid - 1] + sorted_arr[mid]) / 2
    else:
        return sorted_arr[mid]


@jit(nopython=True, cache=True)
def calc_percentile(arr: ArrayF64, percentile: float) -> float:
    """Calculate percentile of an array."""
    n = len(arr)
    if n == 0:
        return 0.0

    sorted_arr = np.sort(arr)
    idx = int(n * percentile / 100)
    if idx >= n:
        idx = n - 1

    return sorted_arr[idx]


@jit(nopython=True, cache=True)
def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default for zero denominator."""
    if denominator == 0.0:
        return default
    return numerator / denominator


@jit(nopython=True, cache=True)
def calc_entropy(hist: ArrayF64) -> float:
    """Calculate entropy from histogram."""
    total = 0.0
    for h in hist:
        total += h

    if total == 0:
        return 0.0

    entropy = 0.0
    for h in hist:
        if h > 0:
            prob = h / total
            entropy -= prob * np.log(prob)

    return entropy


@jit(nopython=True, cache=True)
def calc_herfindahl_index(weights: ArrayF64) -> float:
    """Calculate Herfindahl-Hirschman Index for concentration."""
    total = 0.0
    for w in weights:
        total += w

    if total == 0:
        return 0.0

    hhi = 0.0
    for w in weights:
        share = w / total
        hhi += share * share

    return hhi
