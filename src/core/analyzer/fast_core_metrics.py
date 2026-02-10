"""
Fast Core Metrics - Calculate only the 20 most important metrics.

Target: 50,000+ wallets/sec by computing fewer metrics.

Core metrics:
    1. Total PnL
    2. Total ROI %
    3. Sharpe Ratio
    4. Win Rate %
    5. Profit Factor
    6. Max Drawdown %
    7. Total Volume
    8. Trade Count
    9. Avg Trade Size
    10. Sortino Ratio
    11. Win/Loss Ratio
    12. Expectancy
    13. Trades per Day
    14. Unique Markets
    15. Kelly Criterion
    16. Recovery Factor
    17. Calmar Ratio
    18. Volatility %
    19. Best Market PnL
    20. Quant Score (composite)
"""
import numpy as np
from numba import jit
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class FastCoreData:
    """Minimal wallet data for fast processing."""
    address: str
    n_trades: int
    n_markets: int
    prices: np.ndarray
    sizes: np.ndarray
    fees: np.ndarray
    sides: np.ndarray      # 1=BUY, 0=SELL
    timestamps: np.ndarray
    values: np.ndarray
    market_ids: np.ndarray


@jit(nopython=True, cache=True, fastmath=True)
def calculate_core_20(
    prices: np.ndarray,
    sizes: np.ndarray,
    fees: np.ndarray,
    sides: np.ndarray,
    timestamps: np.ndarray,
    values: np.ndarray,
    market_ids: np.ndarray,
    n_markets: int,
) -> Tuple[float, float, float, float, float, float, float, float, float, float,
           float, float, float, float, float, float, float, float, float, float]:
    """
    Calculate 20 core metrics in a single optimized pass.

    Returns tuple of 20 metrics.
    """
    n = len(prices)
    if n < 2:
        return (0.0,) * 20

    # === BASIC STATS (single pass) ===
    total_volume = 0.0
    total_fees = 0.0
    buy_volume = 0.0
    sell_volume = 0.0

    for i in range(n):
        total_volume += values[i]
        total_fees += fees[i]
        if sides[i] == 1:
            buy_volume += values[i]
        else:
            sell_volume += values[i]

    # Total PnL
    total_pnl = sell_volume - buy_volume - total_fees

    # === PER-MARKET PnL ===
    market_pnl = np.zeros(n_markets, dtype=np.float64)
    market_buy = np.zeros(n_markets, dtype=np.float64)
    market_sell = np.zeros(n_markets, dtype=np.float64)

    for i in range(n):
        mid = market_ids[i]
        if mid < n_markets:
            if sides[i] == 1:
                market_buy[mid] += values[i]
            else:
                market_sell[mid] += values[i]

    winning_pnl = 0.0
    losing_pnl = 0.0
    wins = 0
    losses = 0
    best_pnl = -1e10
    worst_pnl = 1e10

    for m in range(n_markets):
        pnl = market_sell[m] - market_buy[m]
        market_pnl[m] = pnl
        if pnl > 0:
            winning_pnl += pnl
            wins += 1
        elif pnl < 0:
            losing_pnl += abs(pnl)
            losses += 1
        if pnl > best_pnl:
            best_pnl = pnl
        if pnl < worst_pnl:
            worst_pnl = pnl

    # === TIME STATS ===
    time_span = (timestamps[-1] - timestamps[0]) / 86400.0  # days
    if time_span < 1:
        time_span = 1.0

    # === RETURNS & RISK ===
    # Simple per-trade returns
    returns = np.zeros(n - 1, dtype=np.float64)
    for i in range(1, n):
        if values[i - 1] > 0:
            returns[i - 1] = (values[i] - values[i - 1]) / values[i - 1]

    # Mean and std
    mean_return = 0.0
    for r in returns:
        mean_return += r
    mean_return /= len(returns) if len(returns) > 0 else 1

    variance = 0.0
    for r in returns:
        variance += (r - mean_return) ** 2
    variance /= len(returns) if len(returns) > 0 else 1
    volatility = variance ** 0.5

    # Downside volatility
    downside_var = 0.0
    downside_count = 0
    for r in returns:
        if r < 0:
            downside_var += r ** 2
            downside_count += 1
    if downside_count > 0:
        downside_vol = (downside_var / downside_count) ** 0.5
    else:
        downside_vol = volatility

    # === DRAWDOWN ===
    cum_pnl = 0.0
    peak = 0.0
    max_dd = 0.0
    max_dd_pct = 0.0

    for i in range(n):
        if sides[i] == 0:  # SELL
            cum_pnl += values[i]
        else:
            cum_pnl -= values[i]

        if cum_pnl > peak:
            peak = cum_pnl
        dd = peak - cum_pnl
        if dd > max_dd:
            max_dd = dd

    if peak > 0:
        max_dd_pct = max_dd / peak
    else:
        max_dd_pct = 0.0

    # === CALCULATE METRICS ===

    # 1. Total PnL
    m1 = total_pnl

    # 2. Total ROI %
    m2 = (total_pnl / buy_volume * 100) if buy_volume > 0 else 0.0

    # 3. Sharpe Ratio (annualized)
    if volatility > 0:
        m3 = (mean_return * 252 ** 0.5) / volatility
    else:
        m3 = 0.0

    # 4. Win Rate %
    total_markets = wins + losses
    m4 = (wins / total_markets * 100) if total_markets > 0 else 50.0

    # 5. Profit Factor
    m5 = (winning_pnl / losing_pnl) if losing_pnl > 0 else (1.0 if winning_pnl == 0 else 10.0)

    # 6. Max Drawdown %
    m6 = max_dd_pct * 100

    # 7. Total Volume
    m7 = total_volume

    # 8. Trade Count
    m8 = float(n)

    # 9. Avg Trade Size
    m9 = total_volume / n if n > 0 else 0.0

    # 10. Sortino Ratio
    if downside_vol > 0:
        m10 = (mean_return * 252 ** 0.5) / downside_vol
    else:
        m10 = m3  # Fall back to Sharpe

    # 11. Win/Loss Ratio
    avg_win = winning_pnl / wins if wins > 0 else 0.0
    avg_loss = losing_pnl / losses if losses > 0 else 1.0
    m11 = avg_win / avg_loss if avg_loss > 0 else 0.0

    # 12. Expectancy
    win_rate = wins / total_markets if total_markets > 0 else 0.5
    m12 = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    # 13. Trades per Day
    m13 = n / time_span if time_span > 0 else 0.0

    # 14. Unique Markets
    m14 = float(n_markets)

    # 15. Kelly Criterion %
    if m11 > 0:
        kelly = (win_rate - (1 - win_rate) / m11) * 100
        m15 = max(0.0, min(100.0, kelly))
    else:
        m15 = 0.0

    # 16. Recovery Factor
    if max_dd > 0:
        m16 = total_pnl / max_dd
    else:
        m16 = 0.0

    # 17. Calmar Ratio
    if max_dd_pct > 0 and time_span >= 365:
        annualized_return = m2 * (365 / time_span)
        m17 = annualized_return / (max_dd_pct * 100)
    else:
        m17 = 0.0

    # 18. Volatility %
    m18 = volatility * 100

    # 19. Best Market PnL
    m19 = best_pnl if best_pnl > -1e9 else 0.0

    # 20. Quant Score (composite 0-100)
    score = 50.0  # Base

    # ROI contribution
    if m2 > 0:
        score += min(20, m2 / 5)
    else:
        score += max(-20, m2 / 5)

    # Sharpe contribution
    score += min(15, max(-15, m3 * 5))

    # Win rate contribution
    score += (m4 - 50) / 5

    # Drawdown penalty
    score -= min(15, m6 / 5)

    m20 = max(0, min(100, score))

    return (m1, m2, m3, m4, m5, m6, m7, m8, m9, m10,
            m11, m12, m13, m14, m15, m16, m17, m18, m19, m20)


CORE_METRIC_NAMES = {
    1: "Total PnL",
    2: "Total ROI %",
    3: "Sharpe Ratio",
    4: "Win Rate %",
    5: "Profit Factor",
    6: "Max Drawdown %",
    7: "Total Volume",
    8: "Trade Count",
    9: "Avg Trade Size",
    10: "Sortino Ratio",
    11: "Win/Loss Ratio",
    12: "Expectancy",
    13: "Trades per Day",
    14: "Unique Markets",
    15: "Kelly Criterion %",
    16: "Recovery Factor",
    17: "Calmar Ratio",
    18: "Volatility %",
    19: "Best Market PnL",
    20: "Quant Score",
}


def calculate_core_metrics(data: FastCoreData) -> Dict[int, float]:
    """
    Calculate 20 core metrics and return as dict.

    Args:
        data: FastCoreData with numpy arrays

    Returns:
        Dict mapping metric_id (1-20) -> value
    """
    result = calculate_core_20(
        data.prices,
        data.sizes,
        data.fees,
        data.sides,
        data.timestamps,
        data.values,
        data.market_ids,
        data.n_markets,
    )

    return {i + 1: v for i, v in enumerate(result)}


def get_quant_score(data: FastCoreData) -> float:
    """Get just the quant score (metric 20)."""
    result = calculate_core_20(
        data.prices,
        data.sizes,
        data.fees,
        data.sides,
        data.timestamps,
        data.values,
        data.market_ids,
        data.n_markets,
    )
    return result[19]  # Index 19 = metric 20
