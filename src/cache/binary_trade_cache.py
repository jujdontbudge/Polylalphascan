"""
Binary Trade Cache - Ultra-fast NumPy array storage.

Stores trade data as pre-computed NumPy binary arrays for instant loading.
Target: 50,000+ wallets/sec processing speed.

Storage format:
    data/binary_cache/
        {wallet_address}.npz  - Compressed NumPy arrays

Each .npz file contains:
    - prices: float64 array
    - sizes: float64 array
    - fees: float64 array
    - sides: int64 array (1=BUY, 0=SELL)
    - timestamps: int64 array
    - values: float64 array
    - market_ids: int64 array
    - n_markets: int scalar
    - last_timestamp: float scalar (for incremental sync)
"""
import numpy as np
import logging
import os
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import struct

logger = logging.getLogger(__name__)


@dataclass
class BinaryCacheStats:
    """Cache statistics."""
    total_wallets: int
    total_trades: int
    cache_size_mb: float


class BinaryTradeCache:
    """
    Ultra-fast binary cache for trade data.

    Stores NumPy arrays directly as .npz files for instant loading.
    Eliminates Python dict→array conversion overhead.

    Usage:
        cache = BinaryTradeCache()

        # Save trades (converts to numpy once)
        cache.save_wallet(address, trades)

        # Load instantly as numpy arrays
        arrays = cache.load_wallet(address)  # Returns dict of numpy arrays

        # Batch load many wallets
        all_data = cache.load_batch(addresses)  # {addr: arrays}
    """

    def __init__(self, cache_dir: str = "data/binary_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        # Memory cache for hot wallets
        self._memory_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._max_memory_cache = 1000  # Keep up to 1000 wallets in RAM

    def _wallet_path(self, address: str) -> Path:
        """Get file path for wallet."""
        addr = address.lower()
        # Use first 2 chars as subdirectory to avoid too many files in one dir
        subdir = self.cache_dir / addr[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{addr}.npz"

    def has_wallet(self, address: str) -> bool:
        """Check if wallet is cached."""
        return self._wallet_path(address.lower()).exists()

    def save_wallet(self, address: str, trades: List[Any]) -> bool:
        """
        Save trades as binary numpy arrays.

        Args:
            address: Wallet address
            trades: List of trade dicts or objects

        Returns:
            True if saved successfully
        """
        if not trades:
            return False

        address = address.lower()
        n = len(trades)

        # Pre-allocate arrays
        prices = np.zeros(n, dtype=np.float64)
        sizes = np.zeros(n, dtype=np.float64)
        fees = np.zeros(n, dtype=np.float64)
        sides = np.zeros(n, dtype=np.int64)
        timestamps = np.zeros(n, dtype=np.int64)
        values = np.zeros(n, dtype=np.float64)

        # Build market ID mapping
        market_ids_raw = []
        for t in trades:
            if isinstance(t, dict):
                market_ids_raw.append(t.get('market_id') or t.get('condition_id', ''))
            else:
                market_ids_raw.append(getattr(t, 'market_id', '') or getattr(t, 'condition_id', ''))

        unique_markets = list(set(market_ids_raw))
        market_to_idx = {m: i for i, m in enumerate(unique_markets)}
        market_ids = np.zeros(n, dtype=np.int64)

        # Convert trades to arrays
        for i, t in enumerate(trades):
            if isinstance(t, dict):
                prices[i] = float(t.get('price', 0) or 0)
                sizes[i] = float(t.get('size', 0) or 0)
                fees[i] = float(t.get('fee', 0) or 0)
                sides[i] = 1 if str(t.get('side', '')).upper() == 'BUY' else 0

                ts = t.get('timestamp')
                if hasattr(ts, 'timestamp'):
                    timestamps[i] = int(ts.timestamp())
                elif isinstance(ts, (int, float)):
                    timestamps[i] = int(ts)

                values[i] = float(t.get('value', 0) or 0) or (prices[i] * sizes[i])
                market_ids[i] = market_to_idx.get(market_ids_raw[i], 0)
            else:
                # Object with attributes
                prices[i] = float(getattr(t, 'price', 0) or 0)
                sizes[i] = float(getattr(t, 'size', 0) or 0)
                fees[i] = float(getattr(t, 'fee', 0) or 0)
                sides[i] = 1 if str(getattr(t, 'side', '')).upper() == 'BUY' else 0

                ts = getattr(t, 'timestamp', None)
                if hasattr(ts, 'timestamp'):
                    timestamps[i] = int(ts.timestamp())
                elif isinstance(ts, (int, float)):
                    timestamps[i] = int(ts)

                values[i] = float(getattr(t, 'value', 0) or 0) or (prices[i] * sizes[i])
                market_ids[i] = market_to_idx.get(market_ids_raw[i], 0)

        # Sort by timestamp ascending
        sort_idx = np.argsort(timestamps)
        prices = prices[sort_idx]
        sizes = sizes[sort_idx]
        fees = fees[sort_idx]
        sides = sides[sort_idx]
        timestamps = timestamps[sort_idx]
        values = values[sort_idx]
        market_ids = market_ids[sort_idx]

        # Save as compressed numpy archive
        path = self._wallet_path(address)
        with self._lock:
            np.savez_compressed(
                path,
                prices=prices,
                sizes=sizes,
                fees=fees,
                sides=sides,
                timestamps=timestamps,
                values=values,
                market_ids=market_ids,
                n_markets=np.array([len(unique_markets)]),
                last_timestamp=np.array([timestamps[-1] if len(timestamps) > 0 else 0]),
            )

        # Update memory cache
        self._memory_cache[address] = {
            'prices': prices,
            'sizes': sizes,
            'fees': fees,
            'sides': sides,
            'timestamps': timestamps,
            'values': values,
            'market_ids': market_ids,
            'n_markets': len(unique_markets),
        }

        # Evict old entries if memory cache too large
        if len(self._memory_cache) > self._max_memory_cache:
            # Remove oldest entries (simple FIFO)
            keys = list(self._memory_cache.keys())
            for k in keys[:len(keys) // 2]:
                del self._memory_cache[k]

        return True

    def load_wallet(self, address: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Load wallet data as numpy arrays.

        Returns:
            Dict with keys: prices, sizes, fees, sides, timestamps, values, market_ids, n_markets
            Or None if not cached
        """
        address = address.lower()

        # Check memory cache first
        if address in self._memory_cache:
            return self._memory_cache[address]

        path = self._wallet_path(address)
        if not path.exists():
            return None

        try:
            with np.load(path) as data:
                result = {
                    'prices': data['prices'],
                    'sizes': data['sizes'],
                    'fees': data['fees'],
                    'sides': data['sides'],
                    'timestamps': data['timestamps'],
                    'values': data['values'],
                    'market_ids': data['market_ids'],
                    'n_markets': int(data['n_markets'][0]),
                }

            # Cache in memory
            self._memory_cache[address] = result
            return result

        except Exception as e:
            logger.debug(f"Error loading {address}: {e}")
            return None

    def load_batch(
        self,
        addresses: List[str],
        num_threads: int = 4,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Load multiple wallets in parallel.

        Args:
            addresses: List of wallet addresses
            num_threads: Number of parallel threads

        Returns:
            Dict mapping address -> numpy arrays
        """
        result = {}

        # Check memory cache first
        uncached = []
        for addr in addresses:
            addr_lower = addr.lower()
            if addr_lower in self._memory_cache:
                result[addr_lower] = self._memory_cache[addr_lower]
            else:
                uncached.append(addr_lower)

        if not uncached:
            return result

        # Load uncached from disk in parallel
        def load_one(addr):
            return addr, self.load_wallet(addr)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = list(executor.map(load_one, uncached))

        for addr, data in futures:
            if data is not None:
                result[addr] = data

        return result

    def get_cached_wallets(self) -> List[str]:
        """Get all cached wallet addresses."""
        wallets = []
        for subdir in self.cache_dir.iterdir():
            if subdir.is_dir() and len(subdir.name) == 2:
                for f in subdir.glob("*.npz"):
                    wallets.append(f.stem)
        return wallets

    def get_last_timestamp(self, address: str) -> float:
        """Get last trade timestamp for incremental sync."""
        data = self.load_wallet(address)
        if data and len(data['timestamps']) > 0:
            return float(data['timestamps'][-1])
        return 0.0

    def delete_wallet(self, address: str):
        """Remove wallet from cache."""
        address = address.lower()
        path = self._wallet_path(address)
        if path.exists():
            path.unlink()
        if address in self._memory_cache:
            del self._memory_cache[address]

    def clear(self):
        """Clear all cached data."""
        import shutil
        for subdir in self.cache_dir.iterdir():
            if subdir.is_dir():
                shutil.rmtree(subdir)
        self._memory_cache.clear()

    def stats(self) -> BinaryCacheStats:
        """Get cache statistics."""
        total_wallets = 0
        total_trades = 0
        total_size = 0

        for subdir in self.cache_dir.iterdir():
            if subdir.is_dir() and len(subdir.name) == 2:
                for f in subdir.glob("*.npz"):
                    total_wallets += 1
                    total_size += f.stat().st_size
                    # Count trades from file
                    try:
                        with np.load(f) as data:
                            total_trades += len(data['timestamps'])
                    except:
                        pass

        return BinaryCacheStats(
            total_wallets=total_wallets,
            total_trades=total_trades,
            cache_size_mb=total_size / (1024 * 1024),
        )


def migrate_sqlite_to_binary(
    sqlite_cache_path: str = "data/trade_cache.db",
    binary_cache_dir: str = "data/binary_cache",
) -> int:
    """
    Migrate trades from SQLite cache to binary cache.

    Returns number of wallets migrated.
    """
    from src.cache.trade_cache import TradeCache

    sqlite_cache = TradeCache(db_path=sqlite_cache_path)
    binary_cache = BinaryTradeCache(cache_dir=binary_cache_dir)

    wallets = sqlite_cache.get_cached_wallets()
    migrated = 0

    for addr in wallets:
        trades = sqlite_cache.get_trades_as_dicts(addr)
        if trades:
            binary_cache.save_wallet(addr, trades)
            migrated += 1

    return migrated
