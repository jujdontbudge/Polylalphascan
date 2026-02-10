"""
Ultra-Fast 117 Metrics Calculator using Pure Numba.

Target: 50,000+ wallets/second calculation speed.

This module implements all 117 metrics in pure Numba JIT-compiled functions,
completely bypassing pandas for maximum performance.

Metric Groups:
    A (1-10):   Profitability
    B (11-21):  Risk & Stability
    C (22-32):  Win/Loss Dynamics
    D (33-40):  Activity & Frequency
    E (41-48):  Timing & Speed
    F (49-57):  Money Management
    G (58-66):  Strategy Signature
    H (67-74):  Psychological
    I (75-83):  Advanced Alpha
    J (84-94):  Hygiene & Metadata
    K (95-100): Copy-ability
    L (101-104): Market Intelligence
    M (105-108): Execution Quality
    N (109-113): Bot/Quant Detection
    O (114-117): Portfolio-Level
"""
import numpy as np
from numba import jit, prange
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

from .numba_intermediates import (
    calc_basic_stats,
    calc_market_arrays,
    calc_win_loss_stats,
    calc_streak_stats,
    calc_time_stats,
    calc_per_market_hold_times,
    calc_hold_time_stats,
    calc_per_trade_pnl,
    calc_returns_and_risk,
    calc_drawdown_stats,
    calc_var_metrics,
    calc_median,
    calc_percentile,
    safe_divide,
    calc_entropy,
    calc_herfindahl_index,
    calc_market_buy_volume,
)

# Type aliases
ArrayF64 = np.ndarray
ArrayI64 = np.ndarray


# =============================================================================
# FAST WALLET DATA STRUCTURE
# =============================================================================

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
            sides[i] = 1 if str(t.get('side', '')).upper() == 'BUY' else 0

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
# GROUP A: Profitability Metrics (1-10)
# =============================================================================

@jit(nopython=True, cache=True)
def calc_group_a_profitability(
    market_pnl: ArrayF64,
    total_pnl: float,
    buy_volume: float,
    total_volume: float,
    time_span_days: float,
    n_markets: int,
    realized_pnl: float,
    unrealized_pnl: float,
) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
    """
    Calculate Group A: Profitability Metrics (IDs 1-10).

    Returns tuple of 10 metric values.
    """
    # Metric 1: Total PnL (USD)
    m1 = total_pnl

    # Metric 2: Total ROI (%)
    m2 = safe_divide(total_pnl, buy_volume, 0.0) * 100

    # Metric 3: Annualized ROI (%)
    if time_span_days > 0 and buy_volume > 0:
        roi = total_pnl / buy_volume
        annualized = ((1 + roi) ** (365 / time_span_days) - 1) * 100
        m3 = min(max(annualized, -100), 1000)  # Cap extremes
    else:
        m3 = m2

    # Metric 4: Profit Factor
    winning_pnl = 0.0
    losing_pnl = 0.0
    for pnl in market_pnl:
        if pnl > 0:
            winning_pnl += pnl
        elif pnl < 0:
            losing_pnl += abs(pnl)
    m4 = safe_divide(winning_pnl, losing_pnl, 1.0)

    # Metric 5: Average Market PnL
    m5 = safe_divide(total_pnl, n_markets, 0.0)

    # Metric 6: Median Market PnL
    m6 = calc_median(market_pnl) if n_markets > 0 else 0.0

    # Metric 7: Best Market PnL
    m7 = market_pnl.max() if n_markets > 0 else 0.0

    # Metric 8: Worst Market PnL
    m8 = market_pnl.min() if n_markets > 0 else 0.0

    # Metric 9: PnL per Volume
    m9 = safe_divide(total_pnl, total_volume, 0.0)

    # Metric 10: Realized vs Unrealized Ratio
    total_magnitude = abs(realized_pnl) + abs(unrealized_pnl)
    m10 = safe_divide(realized_pnl, total_magnitude, 1.0)

    return m1, m2, m3, m4, m5, m6, m7, m8, m9, m10


# =============================================================================
# GROUP B: Risk & Stability Metrics (11-21)
# =============================================================================

@jit(nopython=True, cache=True)
def calc_group_b_risk(
    returns: ArrayF64,
    cum_returns: ArrayF64,
    total_pnl: float,
    total_volume: float,
    max_dd: float,
    avg_dd: float,
    max_dd_duration: int,
    volatility: float,
    downside_vol: float,
    sharpe: float,
    sortino: float,
    var_5: float,
    cvar_5: float,
) -> Tuple[float, float, float, float, float, float, float, float, float, float, float]:
    """
    Calculate Group B: Risk & Stability Metrics (IDs 11-21).

    Returns tuple of 11 metric values.
    """
    n = len(returns)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Mean return
    mean_return = 0.0
    for r in returns:
        mean_return += r
    mean_return /= n if n > 0 else 1

    # Metric 11: Sharpe Ratio
    m11 = sharpe

    # Metric 12: Sortino Ratio
    m12 = sortino

    # Metric 13: Calmar Ratio
    annualized_return = mean_return * 252
    m13 = safe_divide(annualized_return, max_dd, 0.0) if max_dd > 0 else 0.0

    # Metric 14: Maximum Drawdown (%)
    m14 = max_dd * 100

    # Metric 15: Average Drawdown (%)
    m15 = avg_dd * 100

    # Metric 16: Max Drawdown Duration
    m16 = float(max_dd_duration)

    # Metric 17: Return Volatility (%)
    m17 = volatility * 100

    # Metric 18: Downside Volatility (%)
    m18 = downside_vol * 100

    # Metric 19: Value at Risk (5%)
    m19 = var_5 * 100

    # Metric 20: Conditional VaR (5%)
    m20 = cvar_5 * 100

    # Metric 21: Recovery Factor
    max_dd_usd = max_dd * total_volume
    m21 = safe_divide(total_pnl, max_dd_usd, 0.0) if max_dd_usd > 0 else 0.0

    return m11, m12, m13, m14, m15, m16, m17, m18, m19, m20, m21


# =============================================================================
# GROUP C: Win/Loss Dynamics Metrics (22-32)
# =============================================================================

@jit(nopython=True, cache=True)
def calc_group_c_winloss(
    wins: int,
    losses: int,
    win_rate: float,
    winning_pnl: float,
    losing_pnl: float,
    max_win_streak: int,
    max_loss_streak: int,
    avg_streak: float,
) -> Tuple[float, float, float, float, float, float, float, float, float, float, float]:
    """
    Calculate Group C: Win/Loss Dynamics Metrics (IDs 22-32).

    Returns tuple of 11 metric values.
    """
    total_trades = wins + losses

    # Metric 22: Win Rate (%)
    m22 = win_rate * 100

    # Metric 23: Loss Rate (%)
    m23 = (1 - win_rate) * 100 if total_trades > 0 else 0.0

    # Metric 24: Average Win (USD)
    m24 = safe_divide(winning_pnl, wins, 0.0)

    # Metric 25: Average Loss (USD)
    m25 = safe_divide(losing_pnl, losses, 0.0)

    # Metric 26: Win/Loss Ratio
    avg_win = m24
    avg_loss = m25 if m25 > 0 else 1.0
    m26 = safe_divide(avg_win, avg_loss, 0.0)

    # Metric 27: Max Consecutive Wins
    m27 = float(max_win_streak)

    # Metric 28: Max Consecutive Losses
    m28 = float(max_loss_streak)

    # Metric 29: Average Streak Length
    m29 = avg_streak

    # Metric 30: Trade Expectancy
    loss_rate = 1 - win_rate
    m30 = (win_rate * avg_win) - (loss_rate * avg_loss)

    # Metric 31: Kelly Criterion (%)
    if m26 > 0:
        kelly = (win_rate - (1 - win_rate) / m26) * 100
        m31 = max(0, min(kelly, 100))
    else:
        m31 = 0.0

    # Metric 32: Break-Even Win Rate (%)
    if m26 > 0:
        m32 = (1 / (1 + m26)) * 100
    else:
        m32 = 50.0

    return m22, m23, m24, m25, m26, m27, m28, m29, m30, m31, m32


# =============================================================================
# GROUP D: Activity & Frequency Metrics (33-40)
# =============================================================================

@jit(nopython=True, cache=True)
def calc_group_d_activity(
    n_trades: int,
    n_markets: int,
    total_volume: float,
    time_span_days: float,
    market_volume: ArrayF64,
    timestamps: ArrayI64,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    Calculate Group D: Activity & Frequency Metrics (IDs 33-40).

    Returns tuple of 8 metric values.
    """
    # Metric 33: Total Trade Count
    m33 = float(n_trades)

    # Metric 34: Total Volume (USD)
    m34 = total_volume

    # Metric 35: Average Trade Size (USD)
    m35 = safe_divide(total_volume, n_trades, 0.0)

    # Metric 36: Trades per Day
    m36 = safe_divide(n_trades, time_span_days, 0.0) if time_span_days > 0 else 0.0

    # Metric 37: Active Trading Days
    # Count unique days from timestamps
    if len(timestamps) > 0:
        sorted_ts = np.sort(timestamps)
        active_days = 1
        current_day = sorted_ts[0] // 86400
        for ts in sorted_ts[1:]:
            day = ts // 86400
            if day != current_day:
                active_days += 1
                current_day = day
        m37 = float(active_days)
    else:
        m37 = 0.0

    # Metric 38: Activity Ratio (%)
    m38 = safe_divide(m37, time_span_days, 0.0) * 100 if time_span_days > 0 else 0.0

    # Metric 39: Unique Markets Traded
    m39 = float(n_markets)

    # Metric 40: Market Concentration (%)
    if n_markets > 0 and total_volume > 0:
        max_market_volume = market_volume.max()
        m40 = (max_market_volume / total_volume) * 100
    else:
        m40 = 0.0

    return m33, m34, m35, m36, m37, m38, m39, m40


# =============================================================================
# GROUP E: Timing & Speed Metrics (41-48)
# =============================================================================

@jit(nopython=True, cache=True)
def calc_group_e_timing(
    avg_hold_hours: float,
    median_hold_hours: float,
    min_hold_seconds: float,
    max_hold_days: float,
    trade_pnl: ArrayF64,
    hold_times: ArrayF64,
    hold_count: int,
    prices: ArrayF64,
    sides: ArrayI64,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    Calculate Group E: Timing & Speed Metrics (IDs 41-48).

    Returns tuple of 8 metric values.
    """
    # Metric 41: Average Hold Time (hours)
    m41 = avg_hold_hours

    # Metric 42: Median Hold Time (hours)
    m42 = median_hold_hours

    # Metric 43: Minimum Hold Time (seconds)
    m43 = min_hold_seconds

    # Metric 44: Maximum Hold Time (days)
    m44 = max_hold_days

    # Metric 45: Avg Time to Profit (hours)
    # Metric 46: Avg Time to Loss (hours)
    win_time_sum = 0.0
    win_count = 0
    loss_time_sum = 0.0
    loss_count = 0

    # Simplified: use trade PnL and hold times (assuming alignment)
    min_len = min(len(trade_pnl), hold_count)
    for i in range(min_len):
        if trade_pnl[i] > 0:
            win_time_sum += hold_times[i]
            win_count += 1
        elif trade_pnl[i] < 0:
            loss_time_sum += hold_times[i]
            loss_count += 1

    m45 = safe_divide(win_time_sum, win_count, 0.0)
    m46 = safe_divide(loss_time_sum, loss_count, 0.0)

    # Metric 47: Entry Timing Score (0-100)
    # BUY at low price = good, SELL at high price = good
    n = len(prices)
    if n > 0:
        score_sum = 0.0
        for i in range(n):
            if sides[i] == 1:  # BUY - lower price is better
                score_sum += (1 - prices[i]) * 100
            else:  # SELL - higher price is better
                score_sum += prices[i] * 100
        m47 = score_sum / n
    else:
        m47 = 50.0

    # Metric 48: Exit Timing Score (0-100)
    # Based on PnL performance
    if n > 0:
        positive_pnl_count = 0
        for pnl in trade_pnl:
            if pnl > 0:
                positive_pnl_count += 1
        win_rate = positive_pnl_count / n

        # Combine with average PnL ratio
        avg_pnl = 0.0
        for pnl in trade_pnl:
            avg_pnl += pnl
        avg_pnl /= n

        # Normalize
        m48 = min(100, max(0, win_rate * 100))
    else:
        m48 = 50.0

    return m41, m42, m43, m44, m45, m46, m47, m48


# =============================================================================
# GROUP F: Money Management Metrics (49-57)
# =============================================================================

@jit(nopython=True, cache=True)
def calc_group_f_money(
    values: ArrayF64,
    prices: ArrayF64,
    sides: ArrayI64,
    total_volume: float,
    total_pnl: float,
    market_volume: ArrayF64,
    n_markets: int,
) -> Tuple[float, float, float, float, float, float, float, float, float]:
    """
    Calculate Group F: Money Management Metrics (IDs 49-57).

    Returns tuple of 9 metric values.
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0

    # Estimate portfolio value
    estimated_portfolio = total_volume / 10 if total_volume > 0 else 1.0

    # Position sizes as % of portfolio
    position_sizes = np.zeros(n, dtype=np.float64)
    for i in range(n):
        position_sizes[i] = (values[i] / estimated_portfolio) * 100

    # Metric 49: Average Position Size (%)
    m49 = 0.0
    for ps in position_sizes:
        m49 += ps
    m49 /= n

    # Metric 50: Max Position Size (%)
    m50 = position_sizes.max()

    # Metric 51: Position Size Std Dev
    if n > 1:
        mean_ps = m49
        var_sum = 0.0
        for ps in position_sizes:
            var_sum += (ps - mean_ps) ** 2
        m51 = (var_sum / n) ** 0.5
    else:
        m51 = 0.0

    # Metric 52: Avg Risk per Trade (%)
    risk_sum = 0.0
    for i in range(n):
        if sides[i] == 1:  # BUY
            risk_sum += values[i] * (1 - prices[i])
        else:  # SELL
            risk_sum += values[i] * prices[i]
    avg_risk = risk_sum / n
    m52 = (avg_risk / estimated_portfolio) * 100

    # Metric 53: Average Leverage (default 1x for Polymarket)
    m53 = 1.0

    # Metric 54: Maximum Leverage (default 1x)
    m54 = 1.0

    # Metric 55: Diversification Score (1 - HHI) * 100
    hhi = calc_herfindahl_index(market_volume)
    m55 = (1 - hhi) * 100

    # Metric 56: Concentration Risk (%) - Top 3 markets
    if n_markets > 0 and total_volume > 0:
        sorted_volumes = np.sort(market_volume)[::-1]  # Descending
        top_3_sum = 0.0
        for i in range(min(3, len(sorted_volumes))):
            top_3_sum += sorted_volumes[i]
        m56 = (top_3_sum / total_volume) * 100
    else:
        m56 = 0.0

    # Metric 57: Capital Efficiency
    avg_capital = total_volume / n if n > 0 else 1.0
    m57 = safe_divide(total_pnl, avg_capital, 0.0)

    return m49, m50, m51, m52, m53, m54, m55, m56, m57


# =============================================================================
# GROUP G: Strategy Signature Metrics (58-66)
# =============================================================================

@jit(nopython=True, cache=True)
def calc_group_g_strategy(
    prices: ArrayF64,
    sides: ArrayI64,
    timestamps: ArrayI64,
    market_ids: ArrayI64,
    values: ArrayF64,
    buy_volume: float,
    sell_volume: float,
    total_volume: float,
    n_markets: int,
    sorted_indices: ArrayI64,
    intervals: ArrayF64,
) -> Tuple[float, float, float, float, float, float, float, float, float]:
    """
    Calculate Group G: Strategy Signature Metrics (IDs 58-66).

    Args:
        sorted_indices: Pre-computed np.argsort(timestamps) for efficiency
        intervals: Pre-computed time intervals between consecutive sorted trades

    Returns tuple of 9 metric values.
    """
    n = len(prices)
    if n == 0:
        return 1.0, 0.0, 50.0, 50.0, 50.0, 0.0, 0.0, 0.0, 50.0

    # Metric 58: Buy/Sell Ratio
    m58 = safe_divide(buy_volume, sell_volume, 1.0)

    # Metric 59: Long/Short Bias (-1 to 1)
    m59 = safe_divide(buy_volume - sell_volume, total_volume, 0.0)

    # Metric 60: Contrarian Score (0-100)
    # Buy when price falling, sell when price rising
    contrarian_count = 0
    total_count = 0

    # sorted_indices passed in from caller

    for idx in range(1, n):
        i = sorted_indices[idx]
        prev_i = sorted_indices[idx - 1]

        if market_ids[i] == market_ids[prev_i]:
            price_up = prices[i] > prices[prev_i]
            if (sides[i] == 1 and not price_up) or (sides[i] == 0 and price_up):
                contrarian_count += 1
            total_count += 1

    m60 = safe_divide(contrarian_count, total_count, 0.5) * 100

    # Metric 61: Momentum Score (inverse of contrarian)
    m61 = 100 - m60

    # Metric 62: Mean Reversion Score
    # Buy after drops, sell after rises (look at 2 trades back)
    reversion_count = 0
    reversion_total = 0

    for idx in range(2, n):
        i = sorted_indices[idx]
        prev_i = sorted_indices[idx - 1]
        prev2_i = sorted_indices[idx - 2]

        if market_ids[i] == market_ids[prev_i] == market_ids[prev2_i]:
            recent_up = prices[prev_i] > prices[prev2_i]
            if (sides[i] == 1 and not recent_up) or (sides[i] == 0 and recent_up):
                reversion_count += 1
            reversion_total += 1

    m62 = safe_divide(reversion_count, reversion_total, 0.5) * 100

    # Metric 63: Arbitrage Activity (%)
    # Quick round-trip trades (< 5 minutes, different sides)
    arb_count = 0
    for idx in range(1, n):
        i = sorted_indices[idx]
        prev_i = sorted_indices[idx - 1]

        if market_ids[i] == market_ids[prev_i]:
            time_diff = timestamps[i] - timestamps[prev_i]
            sides_differ = sides[i] != sides[prev_i]
            if time_diff < 300 and sides_differ:
                arb_count += 1

    m63 = safe_divide(arb_count, n, 0.0) * 100

    # Metric 64: Market Making Ratio (%)
    # Markets with both buy and sell activity
    market_has_buy = np.zeros(n_markets, dtype=np.int64)
    market_has_sell = np.zeros(n_markets, dtype=np.int64)

    for i in range(n):
        m = market_ids[i]
        if sides[i] == 1:
            market_has_buy[m] = 1
        else:
            market_has_sell[m] = 1

    mm_markets = 0
    for m in range(n_markets):
        if market_has_buy[m] == 1 and market_has_sell[m] == 1:
            mm_markets += 1

    m64 = safe_divide(mm_markets, n_markets, 0.0) * 100

    # Metric 65: Event Trading Ratio (%) - placeholder
    m65 = 0.0

    # Metric 66: Strategy Consistency
    # Based on trade size and timing variance
    if n > 4:
        # Size consistency
        mean_size = total_volume / n
        size_var = 0.0
        for v in values:
            size_var += (v - mean_size) ** 2
        size_cv = (size_var / n) ** 0.5 / mean_size if mean_size > 0 else 1.0
        size_score = max(0, 100 - size_cv * 100)

        # Time consistency (use pre-computed intervals)
        n_intervals = len(intervals)
        if n_intervals > 0:
            mean_interval = 0.0
            for interval in intervals:
                mean_interval += interval
            mean_interval /= n_intervals

            if mean_interval > 0:
                interval_var = 0.0
                for interval in intervals:
                    interval_var += (interval - mean_interval) ** 2
                time_cv = (interval_var / n_intervals) ** 0.5 / mean_interval
                time_score = max(0, 100 - time_cv * 50)
            else:
                time_score = 50.0
        else:
            time_score = 50.0

        m66 = (size_score + time_score) / 2
    else:
        m66 = 50.0

    return m58, m59, m60, m61, m62, m63, m64, m65, m66


# =============================================================================
# GROUP H: Psychological Metrics (67-74)
# =============================================================================

@jit(nopython=True, cache=True)
def calc_group_h_psychological(
    prices: ArrayF64,
    sides: ArrayI64,
    timestamps: ArrayI64,
    market_ids: ArrayI64,
    values: ArrayF64,
    trade_pnl: ArrayF64,
    n_trades: int,
    time_span_days: float,
    sorted_indices: ArrayI64,
    values_mean: float,
    values_std: float,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    Calculate Group H: Psychological Metrics (IDs 67-74).

    Args:
        sorted_indices: Pre-computed np.argsort(timestamps) for efficiency
        values_mean: Pre-computed mean of values array
        values_std: Pre-computed std of values array

    Returns tuple of 8 metric values.
    """
    n = len(prices)
    if n < 5:
        return 50.0, 0.0, 0.0, 1.0, 0.0, 0.0, 50.0, 50.0

    # sorted_indices passed in from caller

    # Metric 67: Disposition Effect (0-100)
    # Holding losers longer than winners
    winner_times = np.zeros(n, dtype=np.float64)
    winner_count = 0
    loser_times = np.zeros(n, dtype=np.float64)
    loser_count = 0

    for idx in range(1, n):
        i = sorted_indices[idx]
        prev_i = sorted_indices[idx - 1]

        if market_ids[i] == market_ids[prev_i]:
            time_diff = (timestamps[i] - timestamps[prev_i]) / 3600.0
            pnl = trade_pnl[i]

            if pnl > 0:
                winner_times[winner_count] = time_diff
                winner_count += 1
            elif pnl < 0:
                loser_times[loser_count] = time_diff
                loser_count += 1

    if winner_count > 0 and loser_count > 0:
        avg_winner = 0.0
        for i in range(winner_count):
            avg_winner += winner_times[i]
        avg_winner /= winner_count

        avg_loser = 0.0
        for i in range(loser_count):
            avg_loser += loser_times[i]
        avg_loser /= loser_count

        ratio = safe_divide(avg_loser - avg_winner, avg_winner + avg_loser, 0.0)
        m67 = (ratio + 1) * 50
    else:
        m67 = 50.0

    # Metric 68: Revenge Trading Score (0-100)
    # Faster trading after losses
    time_after_win = np.zeros(n, dtype=np.float64)
    win_count = 0
    time_after_loss = np.zeros(n, dtype=np.float64)
    loss_count = 0

    for idx in range(n - 1):
        i = sorted_indices[idx]
        next_i = sorted_indices[idx + 1]

        time_to_next = timestamps[next_i] - timestamps[i]

        if trade_pnl[i] > 0:
            time_after_win[win_count] = time_to_next
            win_count += 1
        elif trade_pnl[i] < 0:
            time_after_loss[loss_count] = time_to_next
            loss_count += 1

    if win_count > 0 and loss_count > 0:
        avg_after_win = 0.0
        for i in range(win_count):
            avg_after_win += time_after_win[i]
        avg_after_win /= win_count

        avg_after_loss = 0.0
        for i in range(loss_count):
            avg_after_loss += time_after_loss[i]
        avg_after_loss /= loss_count

        if avg_after_loss < avg_after_win:
            ratio = safe_divide(avg_after_win - avg_after_loss, avg_after_win, 0.0)
            m68 = min(100, ratio * 100)
        else:
            m68 = 0.0
    else:
        m68 = 0.0

    # Metric 69: Overconfidence Index (0-100)
    # Position size increase after wins
    size_after_win = np.zeros(n, dtype=np.float64)
    size_win_count = 0

    for idx in range(n - 1):
        i = sorted_indices[idx]
        next_i = sorted_indices[idx + 1]

        if trade_pnl[i] > 0:
            size_after_win[size_win_count] = values[next_i]
            size_win_count += 1

    if size_win_count > 0:
        avg_size_after_win = 0.0
        for i in range(size_win_count):
            avg_size_after_win += size_after_win[i]
        avg_size_after_win /= size_win_count

        baseline = 0.0
        for v in values:
            baseline += v
        baseline /= n

        if avg_size_after_win > baseline:
            ratio = safe_divide(avg_size_after_win - baseline, baseline, 0.0)
            m69 = min(100, ratio * 100)
        else:
            m69 = 0.0
    else:
        m69 = 0.0

    # Metric 70: Loss Aversion Ratio
    size_change_after_win = np.zeros(n, dtype=np.float64)
    scaw_count = 0
    size_change_after_loss = np.zeros(n, dtype=np.float64)
    scal_count = 0

    for idx in range(n - 1):
        i = sorted_indices[idx]
        next_i = sorted_indices[idx + 1]

        size_change = abs(values[next_i] - values[i])

        if trade_pnl[i] > 0:
            size_change_after_win[scaw_count] = size_change
            scaw_count += 1
        elif trade_pnl[i] < 0:
            size_change_after_loss[scal_count] = size_change
            scal_count += 1

    if scaw_count > 0 and scal_count > 0:
        avg_change_win = 0.0
        for i in range(scaw_count):
            avg_change_win += size_change_after_win[i]
        avg_change_win /= scaw_count

        avg_change_loss = 0.0
        for i in range(scal_count):
            avg_change_loss += size_change_after_loss[i]
        avg_change_loss /= scal_count

        m70 = safe_divide(avg_change_loss, avg_change_win, 1.0)
    else:
        m70 = 1.0

    # Metric 71: FOMO Index (0-100)
    # Buying after large price increases
    fomo_count = 0
    fomo_total = 0

    for idx in range(1, n):
        i = sorted_indices[idx]
        prev_i = sorted_indices[idx - 1]

        if sides[i] == 1 and market_ids[i] == market_ids[prev_i]:
            fomo_total += 1
            if prices[i] > prices[prev_i] * 1.1:
                fomo_count += 1

    m71 = safe_divide(fomo_count, fomo_total, 0.0) * 100

    # Metric 72: Panic Sell Frequency (0-100)
    panic_count = 0
    sell_count = 0

    for idx in range(1, n):
        i = sorted_indices[idx]
        prev_i = sorted_indices[idx - 1]

        if sides[i] == 0 and market_ids[i] == market_ids[prev_i]:
            sell_count += 1
            if prices[i] < prices[prev_i] * 0.9:
                panic_count += 1

    m72 = safe_divide(panic_count, sell_count, 0.0) * 100

    # Metric 73: Patience Score (0-100)
    trades_per_day = n_trades / time_span_days if time_span_days > 0 else 0

    if trades_per_day <= 1:
        m73 = 90.0
    elif trades_per_day <= 3:
        m73 = 75.0
    elif trades_per_day <= 5:
        m73 = 60.0
    elif trades_per_day <= 10:
        m73 = 40.0
    else:
        m73 = max(20, 100 - trades_per_day * 5)

    # Metric 74: Discipline Score (0-100)
    # Consistent position sizing (use pre-computed values_mean and values_std)
    if n > 2 and values_mean > 0:
        cv = safe_divide(values_std, values_mean, 0.0)
        m74 = max(0, min(100, 100 - cv * 50))
    else:
        m74 = 50.0

    return m67, m68, m69, m70, m71, m72, m73, m74


# =============================================================================
# GROUP I: Advanced Alpha Metrics (75-83)
# =============================================================================

@jit(nopython=True, cache=True)
def calc_group_i_alpha(
    returns: ArrayF64,
    trade_pnl: ArrayF64,
    values: ArrayF64,
    prices: ArrayF64,
    sides: ArrayI64,
    timestamps: ArrayI64,
    market_ids: ArrayI64,
    volatility: float,
    n_markets: int,
    sorted_indices: ArrayI64,
) -> Tuple[float, float, float, float, float, float, float, float, float]:
    """
    Calculate Group I: Advanced Alpha Metrics (IDs 75-83).

    Args:
        sorted_indices: Pre-computed np.argsort(timestamps) for efficiency

    Returns tuple of 9 metric values.
    """
    n = len(returns)
    if n == 0:
        return 0.0, 0.0, 0.0, 1.0, 0.0, 50.0, 1.0, 50.0, 50.0

    # Mean return
    mean_ret = 0.0
    for r in returns:
        mean_ret += r
    mean_ret /= n

    portfolio_return = mean_ret * n
    market_return = 0.05  # Assumed market return

    # Metric 75: Alpha vs Market
    m75 = (portfolio_return - market_return) * 100

    # Metric 76: Information Ratio
    tracking_error = volatility if volatility > 0.01 else 0.01
    alpha = portfolio_return - market_return
    m76 = safe_divide(alpha, tracking_error, 0.0)

    # Metric 77: Treynor Ratio
    risk_free = 0.02
    beta = 0.5 * (volatility / 0.02) if volatility > 0 else 1.0
    beta = max(0.1, min(3.0, beta))
    excess_return = portfolio_return - risk_free
    m77 = safe_divide(excess_return, beta, 0.0)

    # Metric 78: Beta
    m78 = beta

    # Metric 79: Edge per Trade (bps)
    avg_trade_size = 0.0
    for v in values:
        avg_trade_size += v
    avg_trade_size /= n

    avg_pnl = 0.0
    for pnl in trade_pnl:
        avg_pnl += pnl
    avg_pnl /= n

    m79 = safe_divide(avg_pnl, avg_trade_size, 0.0) * 10000

    # Metric 80: Hit Rate on High Conviction
    # High conviction = above average size
    high_conv_wins = 0
    high_conv_total = 0

    for i in range(n):
        if values[i] > avg_trade_size:
            high_conv_total += 1
            if trade_pnl[i] > 0:
                high_conv_wins += 1

    m80 = safe_divide(high_conv_wins, high_conv_total, 0.5) * 100

    # Metric 81: Asymmetric Payoff Ratio
    pos_sum = 0.0
    pos_count = 0
    neg_sum = 0.0
    neg_count = 0

    for r in returns:
        if r > 0:
            pos_sum += r
            pos_count += 1
        elif r < 0:
            neg_sum += abs(r)
            neg_count += 1

    avg_up = safe_divide(pos_sum, pos_count, 0.0)
    avg_down = safe_divide(neg_sum, neg_count, 0.01)
    m81 = safe_divide(avg_up, avg_down, 1.0)

    # Metric 82: Smart Money Indicator (0-100)
    # Trading at extremes profitably
    # OPTIMIZED: Precompute min/max prices per market (O(n) instead of O(n²))
    market_min_price = np.full(n_markets, 1.0, dtype=np.float64)
    market_max_price = np.zeros(n_markets, dtype=np.float64)

    for i in range(n):
        m = market_ids[i]
        if prices[i] < market_min_price[m]:
            market_min_price[m] = prices[i]
        if prices[i] > market_max_price[m]:
            market_max_price[m] = prices[i]

    smart_trades = 0
    total_trades = 0

    for i in range(n):
        m = market_ids[i]
        min_price = market_min_price[m]
        max_price = market_max_price[m]

        if max_price > min_price:
            is_extreme_low = prices[i] <= min_price * 1.1
            is_extreme_high = prices[i] >= max_price * 0.9

            if (sides[i] == 1 and is_extreme_low) or (sides[i] == 0 and is_extreme_high):
                smart_trades += 1
            total_trades += 1

    m82 = safe_divide(smart_trades, total_trades, 0.5) * 100

    # Metric 83: Early Mover Advantage (0-100)
    # Profit from early trades in markets (sorted_indices passed in from caller)
    early_pnl = 0.0
    total_pnl_sum = 0.0

    for pnl in trade_pnl:
        total_pnl_sum += pnl

    # Consider first 25% of trades per market as "early"
    # Reuse market_trade_count from market arrays already computed
    market_trade_idx = np.zeros(n_markets, dtype=np.int64)
    market_trade_count = np.zeros(n_markets, dtype=np.int64)

    # Count trades per market
    for i in range(n):
        market_trade_count[market_ids[i]] += 1

    # Get early trades
    for idx in range(n):
        i = sorted_indices[idx]
        m = market_ids[i]
        early_threshold = max(1, market_trade_count[m] // 4)

        if market_trade_idx[m] < early_threshold:
            early_pnl += trade_pnl[i]

        market_trade_idx[m] += 1

    m83 = safe_divide(early_pnl, total_pnl_sum, 0.5) * 100 if total_pnl_sum != 0 else 50.0

    return m75, m76, m77, m78, m79, m80, m81, m82, m83


# =============================================================================
# GROUP J: Hygiene & Metadata Metrics (84-94)
# =============================================================================

@jit(nopython=True, cache=True)
def calc_group_j_hygiene(
    timestamps: ArrayI64,
    prices: ArrayF64,
    values: ArrayF64,
    fees: ArrayF64,
    market_ids: ArrayI64,
    current_timestamp: int,
    n_trades: int,
    n_markets: int,
    total_volume: float,
    total_fees: float,
) -> Tuple[float, float, float, float, float, float, float, float, float, float, float]:
    """
    Calculate Group J: Hygiene & Metadata Metrics (IDs 84-94).

    Returns tuple of 11 metric values.
    """
    n = len(timestamps)
    if n == 0:
        return 0.0, 0.0, 0.0, 100.0, 100.0, 0.0, 0.0, 100.0, 0.0, 0.0, 100.0

    sorted_ts = np.sort(timestamps)
    first_trade = sorted_ts[0]
    last_trade = sorted_ts[-1]

    # Metric 84: Wallet Age (days)
    m84 = float(current_timestamp - first_trade) / 86400.0

    # Metric 85: Days Since Last Trade
    m85 = float(current_timestamp - last_trade) / 86400.0

    # Metric 86: Trading History Length (days)
    m86 = float(last_trade - first_trade) / 86400.0

    # Metric 87: Data Completeness (%)
    # Check for valid values
    valid_count = 0
    for i in range(n):
        if timestamps[i] > 0 and prices[i] >= 0 and values[i] > 0:
            valid_count += 1
    m87 = (valid_count / n) * 100

    # Metric 88: Verified Positions (%) - placeholder
    m88 = 100.0

    # Metric 89: Trade Anomaly Rate (%)
    anomalies = 0

    # Size anomalies
    if n >= 10:
        mean_val = total_volume / n
        var_val = 0.0
        for v in values:
            var_val += (v - mean_val) ** 2
        std_val = (var_val / n) ** 0.5

        if std_val > 0:
            for v in values:
                if abs(v - mean_val) > 3 * std_val:
                    anomalies += 1

    # Price anomalies
    for p in prices:
        if p < 0 or p > 1:
            anomalies += 1

    # Time anomalies (< 1 second apart)
    for i in range(1, n):
        if sorted_ts[i] - sorted_ts[i-1] < 1:
            anomalies += 1

    m89 = safe_divide(anomalies, n * 3, 0.0) * 100

    # Metric 90: Failed Trade Rate (%) - placeholder
    m90 = 0.0

    # Metric 91: Gas Efficiency Score (fee efficiency)
    fee_ratio = safe_divide(total_fees, total_volume, 0.0)
    m91 = max(0, 100 - fee_ratio * 1000)

    # Metric 92: Average Slippage (%)
    # Deviation from round prices
    slippage_sum = 0.0
    for p in prices:
        nearest_round = round(p * 10) / 10
        deviation = abs(p - nearest_round) / p if p > 0 else 0
        slippage_sum += deviation
    m92 = (slippage_sum / n) * 100

    # Metric 93: Cross-Market Activity
    # Count markets with overlapping activity (simplified)
    m93 = float(n_markets)

    # Metric 94: Data Quality Score (composite)
    m94 = m87 * 0.3 + m88 * 0.3 + (100 - m89) * 0.2 + m91 * 0.2

    return m84, m85, m86, m87, m88, m89, m90, m91, m92, m93, m94


# =============================================================================
# GROUP K: Copy-ability Metrics (95-100)
# =============================================================================

@jit(nopython=True, cache=True)
def calc_group_k_copyability(
    timestamps: ArrayI64,
    market_ids: ArrayI64,
    values: ArrayF64,
    trade_pnl: ArrayF64,
    time_span_days: float,
    n_trades: int,
    n_markets: int,
    total_volume: float,
    sharpe: float,
    sides: ArrayI64,
    sorted_indices: ArrayI64,
    values_mean: float,
    values_std: float,
) -> Tuple[float, float, float, float, float, float]:
    """
    Calculate Group K: Copy-ability Metrics (IDs 95-100).

    Args:
        sorted_indices: Pre-computed np.argsort(timestamps) for efficiency
        values_mean: Pre-computed mean of values array
        values_std: Pre-computed std of values array

    Returns tuple of 6 metric values.
    """
    n = len(timestamps)
    if n == 0:
        return 0.0, 300.0, 0.0, 50.0, 50.0, 50.0

    trades_per_day = n_trades / time_span_days if time_span_days > 0 else 0

    # Metric 95: Trade Frequency Score
    if trades_per_day < 0.5:
        m95 = 30.0
    elif trades_per_day <= 1:
        m95 = 60.0
    elif trades_per_day <= 3:
        m95 = 90.0
    elif trades_per_day <= 5:
        m95 = 80.0
    elif trades_per_day <= 10:
        m95 = 60.0
    elif trades_per_day <= 20:
        m95 = 40.0
    else:
        m95 = 20.0

    # Metric 96: Execution Window (seconds)
    # Time between trades in same market (sorted_indices passed in from caller)
    windows = np.zeros(n, dtype=np.float64)
    window_count = 0

    for idx in range(1, n):
        i = sorted_indices[idx]
        prev_i = sorted_indices[idx - 1]

        if market_ids[i] == market_ids[prev_i]:
            windows[window_count] = float(timestamps[i] - timestamps[prev_i])
            window_count += 1

    if window_count > 0:
        m96 = calc_median(windows[:window_count])
    else:
        m96 = 300.0

    # Metric 97: Liquidity Requirement
    avg_size = total_volume / n
    m97 = avg_size * 2

    # Metric 98: Copy Scalability Score
    # Similar performance across trade sizes
    if n >= 10:
        quartile_size = n // 4
        if quartile_size >= 2:
            sorted_by_size = np.argsort(values)
            small_wins = 0
            small_total = 0
            large_wins = 0
            large_total = 0

            for i in range(quartile_size):
                idx = sorted_by_size[i]
                small_total += 1
                if trade_pnl[idx] > 0:
                    small_wins += 1

            for i in range(n - quartile_size, n):
                idx = sorted_by_size[i]
                large_total += 1
                if trade_pnl[idx] > 0:
                    large_wins += 1

            small_wr = safe_divide(small_wins, small_total, 0.5)
            large_wr = safe_divide(large_wins, large_total, 0.5)

            wr_diff = abs(large_wr - small_wr)
            m98 = max(0, 100 - wr_diff * 200)
        else:
            m98 = 50.0
    else:
        m98 = 50.0

    # Metric 99: Transparency Score
    # Consistent patterns
    scores = np.zeros(4, dtype=np.float64)

    # Size consistency (use pre-computed values_mean and values_std)
    size_cv = safe_divide(values_std, values_mean, 0.0) if values_mean > 0 else 0.0
    scores[0] = max(0, 100 - size_cv * 50)

    # Time consistency
    if window_count > 0:
        mean_window = 0.0
        for i in range(window_count):
            mean_window += windows[i]
        mean_window /= window_count

        if mean_window > 0:
            var_window = 0.0
            for i in range(window_count):
                var_window += (windows[i] - mean_window) ** 2
            std_window = (var_window / window_count) ** 0.5
            time_cv = safe_divide(std_window, mean_window, 0.0)
            scores[1] = max(0, 100 - time_cv * 30)
        else:
            scores[1] = 50.0
    else:
        scores[1] = 50.0

    # Market focus
    market_volumes = np.zeros(n_markets, dtype=np.float64)
    for i in range(n):
        market_volumes[market_ids[i]] += values[i]

    max_market_vol = market_volumes.max() if n_markets > 0 else 0.0
    top_market_pct = safe_divide(max_market_vol, total_volume, 0.0)

    if 0.3 < top_market_pct < 0.9:
        scores[2] = 80.0
    elif 0.2 < top_market_pct < 0.95:
        scores[2] = 60.0
    else:
        scores[2] = 40.0

    # Direction clarity
    buy_count = 0
    for s in sides:
        if s == 1:
            buy_count += 1
    buy_ratio = buy_count / n
    direction_clarity = abs(buy_ratio - 0.5) * 2
    scores[3] = direction_clarity * 100

    m99 = (scores[0] + scores[1] + scores[2] + scores[3]) / 4

    # Metric 100: Risk-Adjusted Copyability
    sharpe_normalized = min(100, (sharpe + 1) * 25)
    m100 = m99 * 0.3 + m98 * 0.3 + sharpe_normalized * 0.4

    return m95, m96, m97, m98, m99, m100


# =============================================================================
# GROUP L: Market Intelligence Metrics (101-104)
# =============================================================================

@jit(nopython=True, cache=True)
def calc_group_l_market_intel(
    prices: ArrayF64,
    sides: ArrayI64,
    timestamps: ArrayI64,
    market_ids: ArrayI64,
    trade_pnl: ArrayF64,
    values: ArrayF64,
    n_markets: int,
    sorted_indices: ArrayI64,
) -> Tuple[float, float, float, float]:
    """
    Calculate Group L: Market Intelligence Metrics (IDs 101-104).

    Args:
        sorted_indices: Pre-computed np.argsort(timestamps) for efficiency

    Returns tuple of 4 metric values.
    """
    n = len(prices)
    if n < 5:
        return 50.0, 0.0, 50.0, 50.0

    # sorted_indices passed in from caller

    # Metric 101: Market Timing Accuracy (%)
    correct = 0
    total = 0

    for idx in range(n - 1):
        i = sorted_indices[idx]
        next_i = sorted_indices[idx + 1]

        if market_ids[i] == market_ids[next_i]:
            price_went_up = prices[next_i] > prices[i]
            buy_correct = sides[i] == 1 and price_went_up
            sell_correct = sides[i] == 0 and not price_went_up

            if buy_correct or sell_correct:
                correct += 1
            total += 1

    m101 = safe_divide(correct, total, 0.5) * 100

    # Metric 102: Category Expertise Score
    market_pnl = np.zeros(n_markets, dtype=np.float64)
    market_vol = np.zeros(n_markets, dtype=np.float64)

    for i in range(n):
        market_pnl[market_ids[i]] += trade_pnl[i]
        market_vol[market_ids[i]] += values[i]

    if n_markets > 0:
        best_market = 0
        best_pnl = market_pnl[0]
        for m in range(1, n_markets):
            if market_pnl[m] > best_pnl:
                best_pnl = market_pnl[m]
                best_market = m

        best_vol = market_vol[best_market]
        total_vol = 0.0
        for v in market_vol:
            total_vol += v

        roi = safe_divide(best_pnl, best_vol, 0.0)
        vol_conc = safe_divide(best_vol, total_vol, 0.0)

        score = roi * vol_conc * 10000
        m102 = min(100, max(0, score + 50))
    else:
        m102 = 0.0

    # Metric 103: Price Prediction Skill
    pred_correct = 0
    pred_total = 0

    for idx in range(n - 2):
        i = sorted_indices[idx]
        future_i = sorted_indices[idx + 2]

        if market_ids[i] == market_ids[future_i]:
            if sides[i] == 1:  # BUY implies price should go up
                if prices[future_i] > prices[i]:
                    pred_correct += 1
            else:  # SELL implies price should go down
                if prices[future_i] < prices[i]:
                    pred_correct += 1
            pred_total += 1

    m103 = safe_divide(pred_correct, pred_total, 0.5) * 100

    # Metric 104: Event Outcome Accuracy (%)
    # Win rate on high conviction trades (near 0 or 1)
    hc_wins = 0
    hc_total = 0

    for i in range(n):
        if prices[i] < 0.2 or prices[i] > 0.8:
            hc_total += 1
            if trade_pnl[i] > 0:
                hc_wins += 1

    m104 = safe_divide(hc_wins, hc_total, 0.5) * 100

    return m101, m102, m103, m104


# =============================================================================
# GROUP M: Execution Quality Metrics (105-108)
# =============================================================================

@jit(nopython=True, cache=True)
def calc_group_m_execution(
    prices: ArrayF64,
    sides: ArrayI64,
    timestamps: ArrayI64,
    market_ids: ArrayI64,
    values: ArrayF64,
    sorted_indices: ArrayI64,
    intervals: ArrayF64,
) -> Tuple[float, float, float, float]:
    """
    Calculate Group M: Execution Quality Metrics (IDs 105-108).

    Args:
        sorted_indices: Pre-computed np.argsort(timestamps) for efficiency
        intervals: Pre-computed time intervals between consecutive sorted trades

    Returns tuple of 4 metric values.
    """
    n = len(prices)
    if n < 3:
        return 100.0, 0.0, 50.0, 0.0

    # sorted_indices passed in from caller

    # Metric 105: Fill Rate (%)
    # Proxy: fragmented fills (quick same-direction trades)
    fragmented = 0
    for idx in range(1, n):
        i = sorted_indices[idx]
        prev_i = sorted_indices[idx - 1]

        if market_ids[i] == market_ids[prev_i]:
            time_diff = timestamps[i] - timestamps[prev_i]
            if time_diff < 10 and sides[i] == sides[prev_i]:
                fragmented += 1

    frag_rate = fragmented / n
    m105 = max(70, 100 - frag_rate * 50)

    # Metric 106: Price Improvement (bps)
    improvements = np.zeros(n, dtype=np.float64)
    for i in range(n):
        mid_price = 0.5
        if sides[i] == 1:  # BUY
            improvements[i] = (mid_price - prices[i]) / mid_price * 10000
        else:  # SELL
            improvements[i] = (prices[i] - mid_price) / mid_price * 10000

    m106 = 0.0
    for imp in improvements:
        m106 += imp
    m106 /= n

    # Metric 107: Execution Speed Score (use pre-computed intervals)
    median_time = calc_median(intervals) if len(intervals) > 0 else 0.0

    if median_time < 60:
        m107 = 100.0
    elif median_time < 300:
        m107 = 85.0
    elif median_time < 900:
        m107 = 70.0
    elif median_time < 3600:
        m107 = 50.0
    else:
        m107 = max(20, 50 - (median_time - 3600) / 3600 * 10)

    # Metric 108: Market Impact (bps)
    impacts = np.zeros(n, dtype=np.float64)
    impact_count = 0

    for idx in range(n - 1):
        i = sorted_indices[idx]
        next_i = sorted_indices[idx + 1]

        if market_ids[i] == market_ids[next_i]:
            pre_price = prices[i]
            post_price = prices[next_i]
            if pre_price > 0:
                price_change = abs(post_price - pre_price) / pre_price
                impacts[impact_count] = price_change * 10000
                impact_count += 1

    if impact_count > 0:
        m108 = 0.0
        for i in range(impact_count):
            m108 += impacts[i]
        m108 /= impact_count
    else:
        m108 = 0.0

    return m105, m106, m107, m108


# =============================================================================
# GROUP N: Bot/Quant Detection Metrics (109-113)
# =============================================================================

@jit(nopython=True, cache=True)
def calc_group_n_bot_quant(
    timestamps: ArrayI64,
    prices: ArrayF64,
    market_ids: ArrayI64,
    n_trades: int,
    sorted_indices: ArrayI64,
    intervals: ArrayF64,
) -> Tuple[float, float, float, float, float]:
    """
    Calculate Group N: Bot/Quant Detection Metrics (IDs 109-113).

    Args:
        sorted_indices: Pre-computed np.argsort(timestamps) for efficiency
        intervals: Pre-computed time intervals between consecutive sorted trades

    Returns tuple of 5 metric values.
    """
    n = len(timestamps)
    if n < 5:
        return 50.0, 50.0, 50.0, 0.0, 50.0

    # sorted_indices and intervals passed in from caller

    # Metric 109: Trade Interval Entropy (0-100)
    # Low entropy = regular/bot-like
    num_bins = min(20, len(intervals) // 2)
    if num_bins < 2:
        num_bins = 2

    min_int = intervals.min()
    max_int = intervals.max()

    if max_int == min_int:
        m109 = 0.0  # All same interval = bot
    else:
        bin_width = (max_int - min_int) / num_bins
        hist = np.zeros(num_bins, dtype=np.float64)

        for interval in intervals:
            bin_idx = int((interval - min_int) / bin_width)
            if bin_idx >= num_bins:
                bin_idx = num_bins - 1
            hist[bin_idx] += 1

        m109 = calc_entropy(hist)

        # Normalize
        non_zero = 0
        for h in hist:
            if h > 0:
                non_zero += 1

        if non_zero > 1:
            max_entropy = np.log(float(non_zero))
            m109 = (m109 / max_entropy) * 100 if max_entropy > 0 else 0.0
        else:
            m109 = 0.0

    # Metric 110: Round Number Avoidance (%)
    # Generate round numbers for 0-1 range
    fractions = np.array([0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0])
    tolerance = 0.005

    non_round = 0
    for p in prices:
        is_round = False
        for f in fractions:
            if abs(p - f) < tolerance:
                is_round = True
                break
        if not is_round:
            non_round += 1

    m110 = (non_round / n) * 100

    # Metric 111: Latency Consistency (0-100)
    # Low CV = consistent = bot-like (lower score = more bot)
    mean_int = 0.0
    for interval in intervals:
        mean_int += interval
    mean_int /= len(intervals)

    if mean_int > 0:
        var_int = 0.0
        for interval in intervals:
            var_int += (interval - mean_int) ** 2
        std_int = (var_int / len(intervals)) ** 0.5
        cv = std_int / mean_int
        m111 = min(100, cv * 50)
    else:
        m111 = 50.0

    # Metric 112: Multi-Market Simultaneity
    # Trades in different markets within 10 seconds
    time_window = 10
    simultaneous = 0

    for idx in range(n):
        i = sorted_indices[idx]
        for jdx in range(idx + 1, n):
            j = sorted_indices[jdx]
            if timestamps[j] - timestamps[i] > time_window:
                break
            if market_ids[i] != market_ids[j]:
                simultaneous += 1

    m112 = float(min(simultaneous, n * 5))

    # Metric 113: Session Pattern Regularity (0-100)
    # Trades at consistent hours/minutes
    hours = np.zeros(n, dtype=np.int64)
    minutes = np.zeros(n, dtype=np.int64)

    for i in range(n):
        ts = timestamps[i]
        hours[i] = (ts // 3600) % 24
        minutes[i] = (ts // 60) % 60

    # Hour concentration
    hour_counts = np.zeros(24, dtype=np.int64)
    for h in hours:
        hour_counts[h] += 1

    top_3_hours = 0
    for _ in range(3):
        max_val = 0
        max_idx = 0
        for i in range(24):
            if hour_counts[i] > max_val:
                max_val = hour_counts[i]
                max_idx = i
        top_3_hours += max_val
        hour_counts[max_idx] = 0

    hour_score = (top_3_hours / n) * 100

    # Minute concentration (cron jobs run at specific minutes)
    minute_counts = np.zeros(60, dtype=np.int64)
    for m in minutes:
        minute_counts[m] += 1

    top_5_minutes = 0
    for _ in range(5):
        max_val = 0
        max_idx = 0
        for i in range(60):
            if minute_counts[i] > max_val:
                max_val = minute_counts[i]
                max_idx = i
        top_5_minutes += max_val
        minute_counts[max_idx] = 0

    minute_score = (top_5_minutes / n) * 100

    m113 = (hour_score + minute_score) / 2

    return m109, m110, m111, m112, m113


# =============================================================================
# GROUP O: Portfolio-Level Metrics (114-117)
# =============================================================================

@jit(nopython=True, cache=True)
def calc_group_o_portfolio(
    prices: ArrayF64,
    values: ArrayF64,
    sides: ArrayI64,
    market_ids: ArrayI64,
    market_pnl: ArrayF64,
    market_volume: ArrayF64,
    total_volume: float,
    n_markets: int,
    buy_volume: float,
    sell_volume: float,
) -> Tuple[float, float, float, float]:
    """
    Calculate Group O: Portfolio-Level Metrics (IDs 114-117).

    Returns tuple of 4 metric values.
    """
    n = len(prices)
    if n == 0:
        return 0.0, 1.0, 0.0, 0.0

    # Metric 114: Portfolio Heat Score
    # Risk concentration in recent trades
    recent_count = min(10, n)
    recent_value = 0.0
    for i in range(n - recent_count, n):
        recent_value += values[i]

    recency_ratio = safe_divide(recent_value, total_volume, 0.0)

    # Risk positions (price near 0.5 = uncertain)
    risk_value = 0.0
    for i in range(n):
        if 0.3 < prices[i] < 0.7:
            risk_value += values[i]

    risk_ratio = safe_divide(risk_value, total_volume, 0.0)

    # Concentration
    max_market_vol = market_volume.max() if n_markets > 0 else 0.0
    concentration = safe_divide(max_market_vol, total_volume, 0.0)

    m114 = min(100, (concentration * 50 + risk_ratio * 50))

    # Metric 115: Correlation Exposure (beta proxy)
    # Volume variance over time
    if n >= 5:
        # Group by day
        day_volumes = np.zeros(n, dtype=np.float64)
        sorted_indices = np.argsort(values)  # Just use value variation as proxy

        for i in range(n):
            day_volumes[i] = values[i]

        mean_vol = total_volume / n
        var_vol = 0.0
        for v in day_volumes:
            var_vol += (v - mean_vol) ** 2
        std_vol = (var_vol / n) ** 0.5

        vol_cv = safe_divide(std_vol, mean_vol, 0.0)
        m115 = min(2.0, max(0.5, 0.5 + vol_cv * 3))
    else:
        m115 = 1.0

    # Metric 116: Sector Balance Score
    # Gini coefficient of market volumes
    if n_markets > 1 and total_volume > 0:
        weights = np.zeros(n_markets, dtype=np.float64)
        for m in range(n_markets):
            weights[m] = market_volume[m] / total_volume

        sorted_weights = np.sort(weights)
        cumsum = 0.0
        cumsum_total = 0.0
        for i in range(n_markets):
            cumsum += sorted_weights[i]
            cumsum_total += cumsum

        sum_weights = 0.0
        for w in sorted_weights:
            sum_weights += w

        gini = 1 - 2 * cumsum_total / (n_markets * sum_weights) + 1 / n_markets
        m116 = (1 - gini) * 100
    else:
        m116 = 100.0 if n_markets == 1 else 0.0

    # Metric 117: Net Exposure (%)
    net_exposure = abs(buy_volume - sell_volume) / total_volume * 100 if total_volume > 0 else 0.0
    m117 = net_exposure

    return m114, m115, m116, m117


# =============================================================================
# MASTER FUNCTION: Calculate All 117 Metrics
# =============================================================================

@jit(nopython=True, cache=True)
def calculate_all_117_metrics(
    prices: ArrayF64,
    sizes: ArrayF64,
    fees: ArrayF64,
    sides: ArrayI64,
    timestamps: ArrayI64,
    market_ids: ArrayI64,
    values: ArrayF64,
    n_markets: int,
    current_timestamp: int,
) -> ArrayF64:
    """
    Calculate ALL 117 metrics in a single optimized pass.

    Args:
        prices: Trade prices (0-1 for prediction markets)
        sizes: Trade sizes
        fees: Trade fees
        sides: 1=BUY, 0=SELL
        timestamps: Unix timestamps
        market_ids: Market index (0 to n_markets-1)
        values: Trade values (price * size)
        n_markets: Number of unique markets
        current_timestamp: Current Unix timestamp for age calculations

    Returns:
        NumPy array of 117 float64 values (metrics 1-117)
    """
    metrics = np.zeros(117, dtype=np.float64)
    n = len(prices)

    if n == 0:
        return metrics

    # ==========================================================================
    # COMPUTE INTERMEDIATE VALUES
    # ==========================================================================

    # Level 1: Basic stats
    total_volume, buy_volume, sell_volume, buy_count, sell_count, total_fees = \
        calc_basic_stats(values, fees, sides)

    # Level 2: Per-market arrays
    market_pnl, market_volume, market_count, market_cost, market_shares = \
        calc_market_arrays(prices, sizes, fees, sides, market_ids, n_markets)

    total_pnl = 0.0
    for pnl in market_pnl:
        total_pnl += pnl

    # Level 3: Win/loss stats
    _, wins, losses, win_rate, winning_pnl, losing_pnl = calc_win_loss_stats(market_pnl)
    max_win_streak, max_loss_streak, avg_streak = calc_streak_stats(market_pnl)

    # Level 4: Time stats
    intervals, time_span_days, mean_interval, std_interval, median_interval = \
        calc_time_stats(timestamps)

    hold_times, hold_count = calc_per_market_hold_times(timestamps, sides, market_ids, n_markets)
    avg_hold, median_hold, min_hold_sec, max_hold_days = calc_hold_time_stats(hold_times, hold_count)

    # Level 5: Returns and risk
    trade_pnl = calc_per_trade_pnl(prices, sizes, fees, sides, market_ids, n_markets)
    returns, cum_returns, max_dd, volatility, sharpe, downside_vol, sortino = \
        calc_returns_and_risk(trade_pnl, values)
    avg_dd, _, max_dd_duration = calc_drawdown_stats(cum_returns)
    var_5, cvar_5 = calc_var_metrics(returns, 5.0)

    # ==========================================================================
    # SHARED PRECOMPUTED VALUES (Optimization: compute once, use in many groups)
    # ==========================================================================

    # Compute sorted_indices once for all groups that need time-sorted access
    sorted_indices = np.argsort(timestamps)

    # Compute intervals for sorted trades (used by groups G, M, N)
    sorted_intervals = np.zeros(n - 1, dtype=np.float64)
    for idx in range(n - 1):
        sorted_intervals[idx] = float(timestamps[sorted_indices[idx + 1]] - timestamps[sorted_indices[idx]])

    # Compute value statistics once (used by groups H, J, K)
    values_mean = total_volume / n if n > 0 else 0.0
    values_var = 0.0
    for v in values:
        values_var += (v - values_mean) ** 2
    values_std = (values_var / n) ** 0.5 if n > 0 else 0.0

    # ==========================================================================
    # GROUP A: Profitability (1-10)
    # ==========================================================================
    m1, m2, m3, m4, m5, m6, m7, m8, m9, m10 = calc_group_a_profitability(
        market_pnl, total_pnl, buy_volume, total_volume, time_span_days,
        n_markets, total_pnl, 0.0  # realized/unrealized placeholder
    )
    metrics[0] = m1
    metrics[1] = m2
    metrics[2] = m3
    metrics[3] = m4
    metrics[4] = m5
    metrics[5] = m6
    metrics[6] = m7
    metrics[7] = m8
    metrics[8] = m9
    metrics[9] = m10

    # ==========================================================================
    # GROUP B: Risk (11-21)
    # ==========================================================================
    m11, m12, m13, m14, m15, m16, m17, m18, m19, m20, m21 = calc_group_b_risk(
        returns, cum_returns, total_pnl, total_volume,
        max_dd, avg_dd, max_dd_duration, volatility, downside_vol,
        sharpe, sortino, var_5, cvar_5
    )
    metrics[10] = m11
    metrics[11] = m12
    metrics[12] = m13
    metrics[13] = m14
    metrics[14] = m15
    metrics[15] = m16
    metrics[16] = m17
    metrics[17] = m18
    metrics[18] = m19
    metrics[19] = m20
    metrics[20] = m21

    # ==========================================================================
    # GROUP C: Win/Loss (22-32)
    # ==========================================================================
    m22, m23, m24, m25, m26, m27, m28, m29, m30, m31, m32 = calc_group_c_winloss(
        wins, losses, win_rate, winning_pnl, losing_pnl,
        max_win_streak, max_loss_streak, avg_streak
    )
    metrics[21] = m22
    metrics[22] = m23
    metrics[23] = m24
    metrics[24] = m25
    metrics[25] = m26
    metrics[26] = m27
    metrics[27] = m28
    metrics[28] = m29
    metrics[29] = m30
    metrics[30] = m31
    metrics[31] = m32

    # ==========================================================================
    # GROUP D: Activity (33-40)
    # ==========================================================================
    m33, m34, m35, m36, m37, m38, m39, m40 = calc_group_d_activity(
        n, n_markets, total_volume, time_span_days, market_volume, timestamps
    )
    metrics[32] = m33
    metrics[33] = m34
    metrics[34] = m35
    metrics[35] = m36
    metrics[36] = m37
    metrics[37] = m38
    metrics[38] = m39
    metrics[39] = m40

    # ==========================================================================
    # GROUP E: Timing (41-48)
    # ==========================================================================
    m41, m42, m43, m44, m45, m46, m47, m48 = calc_group_e_timing(
        avg_hold, median_hold, min_hold_sec, max_hold_days,
        trade_pnl, hold_times, hold_count, prices, sides
    )
    metrics[40] = m41
    metrics[41] = m42
    metrics[42] = m43
    metrics[43] = m44
    metrics[44] = m45
    metrics[45] = m46
    metrics[46] = m47
    metrics[47] = m48

    # ==========================================================================
    # GROUP F: Money Management (49-57)
    # ==========================================================================
    m49, m50, m51, m52, m53, m54, m55, m56, m57 = calc_group_f_money(
        values, prices, sides, total_volume, total_pnl, market_volume, n_markets
    )
    metrics[48] = m49
    metrics[49] = m50
    metrics[50] = m51
    metrics[51] = m52
    metrics[52] = m53
    metrics[53] = m54
    metrics[54] = m55
    metrics[55] = m56
    metrics[56] = m57

    # ==========================================================================
    # GROUP G: Strategy (58-66)
    # ==========================================================================
    m58, m59, m60, m61, m62, m63, m64, m65, m66 = calc_group_g_strategy(
        prices, sides, timestamps, market_ids, values,
        buy_volume, sell_volume, total_volume, n_markets,
        sorted_indices, sorted_intervals
    )
    metrics[57] = m58
    metrics[58] = m59
    metrics[59] = m60
    metrics[60] = m61
    metrics[61] = m62
    metrics[62] = m63
    metrics[63] = m64
    metrics[64] = m65
    metrics[65] = m66

    # ==========================================================================
    # GROUP H: Psychological (67-74)
    # ==========================================================================
    m67, m68, m69, m70, m71, m72, m73, m74 = calc_group_h_psychological(
        prices, sides, timestamps, market_ids, values, trade_pnl, n, time_span_days,
        sorted_indices, values_mean, values_std
    )
    metrics[66] = m67
    metrics[67] = m68
    metrics[68] = m69
    metrics[69] = m70
    metrics[70] = m71
    metrics[71] = m72
    metrics[72] = m73
    metrics[73] = m74

    # ==========================================================================
    # GROUP I: Alpha (75-83)
    # ==========================================================================
    m75, m76, m77, m78, m79, m80, m81, m82, m83 = calc_group_i_alpha(
        returns, trade_pnl, values, prices, sides, timestamps, market_ids,
        volatility, n_markets, sorted_indices
    )
    metrics[74] = m75
    metrics[75] = m76
    metrics[76] = m77
    metrics[77] = m78
    metrics[78] = m79
    metrics[79] = m80
    metrics[80] = m81
    metrics[81] = m82
    metrics[82] = m83

    # ==========================================================================
    # GROUP J: Hygiene (84-94)
    # ==========================================================================
    m84, m85, m86, m87, m88, m89, m90, m91, m92, m93, m94 = calc_group_j_hygiene(
        timestamps, prices, values, fees, market_ids,
        current_timestamp, n, n_markets, total_volume, total_fees
    )
    metrics[83] = m84
    metrics[84] = m85
    metrics[85] = m86
    metrics[86] = m87
    metrics[87] = m88
    metrics[88] = m89
    metrics[89] = m90
    metrics[90] = m91
    metrics[91] = m92
    metrics[92] = m93
    metrics[93] = m94

    # ==========================================================================
    # GROUP K: Copyability (95-100)
    # ==========================================================================
    m95, m96, m97, m98, m99, m100 = calc_group_k_copyability(
        timestamps, market_ids, values, trade_pnl, time_span_days,
        n, n_markets, total_volume, sharpe, sides,
        sorted_indices, values_mean, values_std
    )
    metrics[94] = m95
    metrics[95] = m96
    metrics[96] = m97
    metrics[97] = m98
    metrics[98] = m99
    metrics[99] = m100

    # ==========================================================================
    # GROUP L: Market Intel (101-104)
    # ==========================================================================
    m101, m102, m103, m104 = calc_group_l_market_intel(
        prices, sides, timestamps, market_ids, trade_pnl, values, n_markets,
        sorted_indices
    )
    metrics[100] = m101
    metrics[101] = m102
    metrics[102] = m103
    metrics[103] = m104

    # ==========================================================================
    # GROUP M: Execution (105-108)
    # ==========================================================================
    m105, m106, m107, m108 = calc_group_m_execution(
        prices, sides, timestamps, market_ids, values,
        sorted_indices, sorted_intervals
    )
    metrics[104] = m105
    metrics[105] = m106
    metrics[106] = m107
    metrics[107] = m108

    # ==========================================================================
    # GROUP N: Bot/Quant (109-113)
    # ==========================================================================
    m109, m110, m111, m112, m113 = calc_group_n_bot_quant(
        timestamps, prices, market_ids, n,
        sorted_indices, sorted_intervals
    )
    metrics[108] = m109
    metrics[109] = m110
    metrics[110] = m111
    metrics[111] = m112
    metrics[112] = m113

    # ==========================================================================
    # GROUP O: Portfolio (114-117)
    # ==========================================================================
    m114, m115, m116, m117 = calc_group_o_portfolio(
        prices, values, sides, market_ids, market_pnl, market_volume,
        total_volume, n_markets, buy_volume, sell_volume
    )
    metrics[113] = m114
    metrics[114] = m115
    metrics[115] = m116
    metrics[116] = m117

    return metrics


# =============================================================================
# HIGH-LEVEL API
# =============================================================================

# Metric ID to index mapping (metric IDs are 1-117, array indices are 0-116)
METRIC_NAMES = {
    # Group A: Profitability
    1: "total_pnl", 2: "total_roi", 3: "annualized_roi", 4: "profit_factor",
    5: "avg_market_pnl", 6: "median_market_pnl", 7: "best_market_pnl",
    8: "worst_market_pnl", 9: "pnl_per_volume", 10: "realized_vs_unrealized",
    # Group B: Risk
    11: "sharpe_ratio", 12: "sortino_ratio", 13: "calmar_ratio",
    14: "max_drawdown", 15: "avg_drawdown", 16: "max_drawdown_duration",
    17: "volatility", 18: "downside_volatility", 19: "var_5",
    20: "cvar_5", 21: "recovery_factor",
    # Group C: Win/Loss
    22: "win_rate", 23: "loss_rate", 24: "avg_win", 25: "avg_loss",
    26: "win_loss_ratio", 27: "max_win_streak", 28: "max_loss_streak",
    29: "avg_streak", 30: "expectancy", 31: "kelly_criterion", 32: "break_even_win_rate",
    # Group D: Activity
    33: "total_trades", 34: "total_volume", 35: "avg_trade_size",
    36: "trades_per_day", 37: "active_days", 38: "activity_ratio",
    39: "unique_markets", 40: "market_concentration",
    # Group E: Timing
    41: "avg_hold_time", 42: "median_hold_time", 43: "min_hold_time",
    44: "max_hold_time", 45: "time_to_profit", 46: "time_to_loss",
    47: "entry_timing_score", 48: "exit_timing_score",
    # Group F: Money
    49: "avg_position_size", 50: "max_position_size", 51: "position_size_std",
    52: "avg_risk_per_trade", 53: "avg_leverage", 54: "max_leverage",
    55: "diversification_score", 56: "concentration_risk", 57: "capital_efficiency",
    # Group G: Strategy
    58: "buy_sell_ratio", 59: "long_short_bias", 60: "contrarian_score",
    61: "momentum_score", 62: "mean_reversion_score", 63: "arbitrage_activity",
    64: "market_making_ratio", 65: "event_trading_ratio", 66: "strategy_consistency",
    # Group H: Psychological
    67: "disposition_effect", 68: "revenge_trading", 69: "overconfidence",
    70: "loss_aversion", 71: "fomo_index", 72: "panic_sell_freq",
    73: "patience_score", 74: "discipline_score",
    # Group I: Alpha
    75: "alpha_vs_market", 76: "information_ratio", 77: "treynor_ratio",
    78: "beta", 79: "edge_per_trade", 80: "high_conviction_hit_rate",
    81: "asymmetric_payoff", 82: "smart_money_indicator", 83: "early_mover_advantage",
    # Group J: Hygiene
    84: "wallet_age", 85: "days_since_last_trade", 86: "trading_history_length",
    87: "data_completeness", 88: "verified_positions", 89: "anomaly_rate",
    90: "failed_trade_rate", 91: "gas_efficiency", 92: "avg_slippage",
    93: "cross_market_activity", 94: "data_quality_score",
    # Group K: Copyability
    95: "frequency_score", 96: "execution_window", 97: "liquidity_requirement",
    98: "scalability_score", 99: "transparency_score", 100: "risk_adjusted_copyability",
    # Group L: Market Intel
    101: "timing_accuracy", 102: "category_expertise", 103: "prediction_skill",
    104: "event_accuracy",
    # Group M: Execution
    105: "fill_rate", 106: "price_improvement", 107: "execution_speed",
    108: "market_impact",
    # Group N: Bot/Quant
    109: "interval_entropy", 110: "round_number_avoidance", 111: "latency_consistency",
    112: "simultaneity", 113: "session_regularity",
    # Group O: Portfolio
    114: "portfolio_heat", 115: "correlation_exposure", 116: "sector_balance",
    117: "net_exposure",
}


def calculate_117_metrics(data: FastWalletData) -> Dict[int, float]:
    """
    Calculate all 117 metrics using ultra-fast Numba implementation.

    Args:
        data: FastWalletData containing trade arrays

    Returns:
        Dict mapping metric_id (1-117) to calculated value
    """
    import time

    if data.n_trades == 0:
        return {i: 0.0 for i in range(1, 118)}

    current_timestamp = int(time.time())

    metrics_array = calculate_all_117_metrics(
        data.prices,
        data.sizes,
        data.fees,
        data.sides,
        data.timestamps,
        data.market_ids,
        data.values,
        max(1, data.n_markets),
        current_timestamp,
    )

    # Convert to dict with metric IDs (1-117)
    return {i + 1: float(metrics_array[i]) for i in range(117)}


def calculate_117_metrics_from_trades(address: str, trades: List[dict]) -> Dict[int, float]:
    """
    Convenience function: calculate all 117 metrics directly from trade list.
    """
    data = FastWalletData.from_trades(address, trades)
    return calculate_117_metrics(data)


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def calculate_batch_117(
    wallets: List[Tuple[str, List[dict]]]
) -> Dict[str, Dict[int, float]]:
    """
    Calculate all 117 metrics for multiple wallets.

    Args:
        wallets: List of (address, trades) tuples

    Returns:
        Dict mapping address -> metrics dict
    """
    results = {}
    for address, trades in wallets:
        results[address] = calculate_117_metrics_from_trades(address, trades)
    return results


# =============================================================================
# PARALLEL BATCH PROCESSING (Numba prange)
# =============================================================================

@jit(nopython=True, parallel=True, cache=True)
def _calculate_batch_metrics_parallel(
    all_prices: np.ndarray,      # Shape: (n_wallets, max_trades)
    all_sizes: np.ndarray,
    all_fees: np.ndarray,
    all_sides: np.ndarray,       # Shape: (n_wallets, max_trades) - int64
    all_timestamps: np.ndarray,  # Shape: (n_wallets, max_trades) - int64
    all_market_ids: np.ndarray,  # Shape: (n_wallets, max_trades) - int64
    all_values: np.ndarray,
    n_trades_per_wallet: np.ndarray,  # Shape: (n_wallets,) - actual trade count
    n_markets_per_wallet: np.ndarray,  # Shape: (n_wallets,) - market count
    current_timestamp: int,
) -> np.ndarray:
    """
    Calculate 117 metrics for multiple wallets in PARALLEL using Numba prange.

    Args:
        all_*: 2D arrays of shape (n_wallets, max_trades) with padded data
        n_trades_per_wallet: Actual trade count per wallet (for slicing)
        n_markets_per_wallet: Number of markets per wallet
        current_timestamp: Current Unix timestamp

    Returns:
        2D array of shape (n_wallets, 117) with all metrics
    """
    n_wallets = all_prices.shape[0]
    results = np.zeros((n_wallets, 117), dtype=np.float64)

    # Parallel loop over wallets
    for w in prange(n_wallets):
        n = n_trades_per_wallet[w]
        n_markets = n_markets_per_wallet[w]

        if n == 0:
            continue

        # Extract this wallet's data (slice to actual trade count)
        prices = all_prices[w, :n]
        sizes = all_sizes[w, :n]
        fees = all_fees[w, :n]
        sides = all_sides[w, :n]
        timestamps = all_timestamps[w, :n]
        market_ids = all_market_ids[w, :n]
        values = all_values[w, :n]

        # Calculate metrics for this wallet
        metrics = calculate_all_117_metrics(
            prices, sizes, fees, sides, timestamps, market_ids, values,
            max(1, n_markets), current_timestamp
        )

        # Copy results
        for i in range(117):
            results[w, i] = metrics[i]

    return results


def prepare_batch_arrays(
    wallets_data: List[FastWalletData],
    max_trades: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare padded arrays for parallel batch processing.

    Args:
        wallets_data: List of FastWalletData objects
        max_trades: Maximum trades to consider per wallet (default: auto-detect)

    Returns:
        Tuple of (all_prices, all_sizes, all_fees, all_sides, all_timestamps,
                  all_market_ids, all_values, n_trades_per_wallet, n_markets_per_wallet)
    """
    n_wallets = len(wallets_data)

    if max_trades is None:
        max_trades = max(w.n_trades for w in wallets_data) if wallets_data else 1

    # Allocate padded arrays
    all_prices = np.zeros((n_wallets, max_trades), dtype=np.float64)
    all_sizes = np.zeros((n_wallets, max_trades), dtype=np.float64)
    all_fees = np.zeros((n_wallets, max_trades), dtype=np.float64)
    all_sides = np.zeros((n_wallets, max_trades), dtype=np.int64)
    all_timestamps = np.zeros((n_wallets, max_trades), dtype=np.int64)
    all_market_ids = np.zeros((n_wallets, max_trades), dtype=np.int64)
    all_values = np.zeros((n_wallets, max_trades), dtype=np.float64)
    n_trades_per_wallet = np.zeros(n_wallets, dtype=np.int64)
    n_markets_per_wallet = np.zeros(n_wallets, dtype=np.int64)

    # Copy data into padded arrays
    for i, wallet in enumerate(wallets_data):
        n = min(wallet.n_trades, max_trades)
        n_trades_per_wallet[i] = n
        n_markets_per_wallet[i] = wallet.n_markets

        if n > 0:
            all_prices[i, :n] = wallet.prices[:n]
            all_sizes[i, :n] = wallet.sizes[:n]
            all_fees[i, :n] = wallet.fees[:n]
            all_sides[i, :n] = wallet.sides[:n]
            all_timestamps[i, :n] = wallet.timestamps[:n]
            all_market_ids[i, :n] = wallet.market_ids[:n]
            all_values[i, :n] = wallet.values[:n]

    return (all_prices, all_sizes, all_fees, all_sides, all_timestamps,
            all_market_ids, all_values, n_trades_per_wallet, n_markets_per_wallet)


def calculate_batch_117_parallel(
    wallets: List[FastWalletData],
    max_trades: int = None,
) -> List[Dict[int, float]]:
    """
    Calculate all 117 metrics for multiple wallets using parallel processing.

    This is significantly faster than sequential processing for batches > 10 wallets.

    Args:
        wallets: List of FastWalletData objects
        max_trades: Maximum trades per wallet to process (default: auto-detect)

    Returns:
        List of metric dictionaries (one per wallet), in same order as input
    """
    import time

    if not wallets:
        return []

    current_timestamp = int(time.time())

    # Prepare padded arrays
    (all_prices, all_sizes, all_fees, all_sides, all_timestamps,
     all_market_ids, all_values, n_trades, n_markets) = prepare_batch_arrays(wallets, max_trades)

    # Run parallel calculation
    results_array = _calculate_batch_metrics_parallel(
        all_prices, all_sizes, all_fees, all_sides, all_timestamps,
        all_market_ids, all_values, n_trades, n_markets, current_timestamp
    )

    # Convert to list of dicts
    results = []
    for w in range(len(wallets)):
        metrics_dict = {i + 1: float(results_array[w, i]) for i in range(117)}
        results.append(metrics_dict)

    return results


def calculate_batch_117_parallel_from_trades(
    wallets: List[Tuple[str, List[dict]]],
    max_trades: int = None,
) -> Dict[str, Dict[int, float]]:
    """
    Calculate all 117 metrics for multiple wallets using parallel processing.

    Convenience function that accepts raw trade data.

    Args:
        wallets: List of (address, trades) tuples
        max_trades: Maximum trades per wallet to process

    Returns:
        Dict mapping address -> metrics dict
    """
    # Convert trades to FastWalletData
    wallet_data = []
    addresses = []
    for address, trades in wallets:
        wallet_data.append(FastWalletData.from_trades(address, trades))
        addresses.append(address)

    # Calculate in parallel
    metrics_list = calculate_batch_117_parallel(wallet_data, max_trades)

    # Build result dict
    return {addr: metrics for addr, metrics in zip(addresses, metrics_list)}


# =============================================================================
# QUANT SCORE CALCULATION
# =============================================================================

def calculate_quant_score(metrics: Dict[int, float]) -> float:
    """
    Calculate overall quant/bot score from Group N metrics.

    Score >= 80: "bot"
    Score >= 50: "quant"
    Score <= 20: "human"
    """
    weights = {
        109: -0.2,   # Entropy (lower = more bot)
        110: 0.25,   # Round number avoidance (higher = more bot)
        111: -0.2,   # Latency consistency (lower = more bot)
        112: 0.2,    # Simultaneity (higher = more bot)
        113: 0.15,   # Session regularity (higher = more bot)
    }

    score = 50  # Start neutral

    for metric_id, weight in weights.items():
        if metric_id in metrics:
            value = metrics[metric_id]

            if weight < 0:
                contribution = (100 - value) * abs(weight)
            else:
                contribution = value * weight

            score += contribution - 50 * abs(weight)

    return max(0, min(100, score))


# =============================================================================
# JIT WARMUP
# =============================================================================

def _warmup():
    """Compile JIT functions on import."""
    dummy = FastWalletData(
        address="0x0",
        n_trades=10,
        n_markets=2,
        prices=np.random.random(10).astype(np.float64),
        sizes=np.random.random(10).astype(np.float64) * 100,
        fees=np.random.random(10).astype(np.float64),
        sides=np.random.randint(0, 2, 10).astype(np.int64),
        timestamps=np.arange(10, dtype=np.int64) * 3600 + 1700000000,
        market_ids=np.random.randint(0, 2, 10).astype(np.int64),
        values=np.random.random(10).astype(np.float64) * 100,
    )
    _ = calculate_117_metrics(dummy)


# Warmup on import
_warmup()
