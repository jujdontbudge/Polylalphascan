"""
Batch Metrics Calculator - Process multiple wallets in parallel using Numba.

Target: 50,000+ wallets/sec by processing batches with prange.
"""
import numpy as np
from numba import jit, prange
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .numba_metrics import FastWalletData, calculate_117_metrics, calculate_quant_score


@jit(nopython=True, parallel=True, cache=True)
def _batch_calculate_basic_stats(
    all_prices: np.ndarray,      # 2D: (n_wallets, max_trades)
    all_sizes: np.ndarray,
    all_fees: np.ndarray,
    all_sides: np.ndarray,
    all_timestamps: np.ndarray,
    all_values: np.ndarray,
    all_market_ids: np.ndarray,
    trade_counts: np.ndarray,    # 1D: (n_wallets,) - actual trades per wallet
    n_markets: np.ndarray,       # 1D: (n_wallets,)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate basic stats for all wallets in parallel.

    Returns:
        total_pnl, total_volume, win_rate, n_trades for each wallet
    """
    n_wallets = len(trade_counts)

    total_pnl = np.zeros(n_wallets, dtype=np.float64)
    total_volume = np.zeros(n_wallets, dtype=np.float64)
    win_rates = np.zeros(n_wallets, dtype=np.float64)

    for i in prange(n_wallets):
        n = trade_counts[i]
        if n < 2:
            continue

        prices = all_prices[i, :n]
        sizes = all_sizes[i, :n]
        values = all_values[i, :n]
        sides = all_sides[i, :n]

        # Calculate volume
        vol = 0.0
        for j in range(n):
            vol += values[j]
        total_volume[i] = vol

        # Simplified PnL (sell - buy)
        buy_value = 0.0
        sell_value = 0.0
        for j in range(n):
            if sides[j] == 1:  # BUY
                buy_value += values[j]
            else:  # SELL
                sell_value += values[j]
        total_pnl[i] = sell_value - buy_value

        # Win rate (simplified: trades where exit price > entry price)
        wins = 0
        total = 0
        for j in range(1, n):
            if prices[j] > prices[j-1]:
                wins += 1
            total += 1
        win_rates[i] = wins / total if total > 0 else 0.5

    return total_pnl, total_volume, win_rates, trade_counts.astype(np.float64)


def batch_calculate_metrics(
    wallet_datas: List[FastWalletData],
    return_full: bool = True,
) -> List[Tuple[Dict[int, float], float]]:
    """
    Calculate metrics for multiple wallets efficiently.

    Uses single-wallet calculation but minimizes Python overhead.

    Args:
        wallet_datas: List of FastWalletData objects
        return_full: If True, return full 117 metrics. If False, return key metrics only.

    Returns:
        List of (metrics_dict, quant_score) tuples
    """
    results = []

    for wd in wallet_datas:
        if wd.n_trades < 2:
            results.append(({}, 0.0))
            continue

        metrics = calculate_117_metrics(wd)
        score = calculate_quant_score(metrics)
        results.append((metrics, score))

    return results


def process_wallets_fast(
    wallet_arrays: Dict[str, Dict[str, np.ndarray]],
    chunk_size: int = 100,
) -> Dict[str, Tuple[Dict[int, float], float]]:
    """
    Process multiple wallets with minimal Python overhead.

    Args:
        wallet_arrays: Dict mapping address -> {prices, sizes, ...}
        chunk_size: Process in chunks for memory efficiency

    Returns:
        Dict mapping address -> (metrics, quant_score)
    """
    results = {}
    addresses = list(wallet_arrays.keys())

    for i in range(0, len(addresses), chunk_size):
        chunk_addrs = addresses[i:i + chunk_size]

        # Convert to FastWalletData and process
        for addr in chunk_addrs:
            data = wallet_arrays[addr]

            if len(data['prices']) < 2:
                results[addr] = ({}, 0.0)
                continue

            wd = FastWalletData(
                address=addr,
                n_trades=len(data['prices']),
                n_markets=data['n_markets'],
                prices=data['prices'],
                sizes=data['sizes'],
                fees=data['fees'],
                sides=data['sides'],
                timestamps=data['timestamps'],
                market_ids=data['market_ids'],
                values=data['values'],
            )

            metrics = calculate_117_metrics(wd)
            score = calculate_quant_score(metrics)
            results[addr] = (metrics, score)

    return results
