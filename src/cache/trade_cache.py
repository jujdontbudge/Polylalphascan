"""
Local Trade Cache - SQLite-based cache for raw trade data.

Eliminates API bottleneck by caching trade data locally:
- First run: Fetch from Goldsky API, save to cache
- Subsequent runs: Load from cache at 50,000+ wallets/sec
- Incremental sync: Only fetch new trades since last sync

Storage:
    - trades: Raw trade data (price, size, fee, side, timestamp, market_id)
    - wallet_sync: Tracking per-wallet sync status and last trade timestamp
"""
import sqlite3
import logging
import threading
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CachedTrade:
    """Trade data from cache."""
    id: str
    wallet_address: str
    timestamp: float  # Unix timestamp
    price: float
    size: float
    fee: float
    side: str  # 'BUY' or 'SELL'
    value: float
    market_id: str
    condition_id: str


@dataclass
class WalletSyncInfo:
    """Sync status for a wallet."""
    address: str
    last_trade_timestamp: float
    last_fetched_at: datetime
    trade_count: int
    is_stale: bool = False


@dataclass
class TradeCacheStats:
    """Cache statistics."""
    total_wallets: int
    total_trades: int
    stale_wallets: int
    cache_size_mb: float
    oldest_sync: Optional[datetime]
    newest_sync: Optional[datetime]


class TradeCache:
    """
    Local SQLite cache for raw trade data.

    Provides fast access to historical trade data without hitting the API.
    Supports incremental sync to keep cache fresh.

    Usage:
        cache = TradeCache()

        # Check if wallet has cached trades
        trades = cache.get_trades("0x123...")

        # Save trades after API fetch
        cache.save_trades("0x123...", trades_from_api)

        # Get wallets that need refresh
        stale = cache.get_stale_wallets(max_age_hours=24)

        # Get trades as numpy arrays (for Numba processing)
        arrays = cache.get_trades_as_numpy("0x123...")
    """

    def __init__(self, db_path: str = "data/trade_cache.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                -- Raw trade data
                CREATE TABLE IF NOT EXISTS trades (
                    id TEXT PRIMARY KEY,
                    wallet_address TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    price REAL NOT NULL,
                    size REAL NOT NULL,
                    fee REAL NOT NULL,
                    side TEXT NOT NULL,
                    value REAL NOT NULL,
                    market_id TEXT,
                    condition_id TEXT,
                    fetched_at TEXT NOT NULL
                );

                -- Indexes for fast queries
                CREATE INDEX IF NOT EXISTS idx_trades_wallet ON trades(wallet_address);
                CREATE INDEX IF NOT EXISTS idx_trades_wallet_ts ON trades(wallet_address, timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp DESC);

                -- Wallet sync tracking
                CREATE TABLE IF NOT EXISTS wallet_sync (
                    address TEXT PRIMARY KEY,
                    last_trade_timestamp REAL DEFAULT 0,
                    last_fetched_at TEXT NOT NULL,
                    trade_count INTEGER DEFAULT 0,
                    first_trade_timestamp REAL DEFAULT 0
                );

                -- Index for finding stale wallets
                CREATE INDEX IF NOT EXISTS idx_wallet_sync_fetched ON wallet_sync(last_fetched_at);
            """)

    # -------------------------------------------------------------------------
    # Core Trade Operations
    # -------------------------------------------------------------------------

    def get_trades(self, address: str) -> List[CachedTrade]:
        """
        Get all cached trades for a wallet.

        Args:
            address: Wallet address

        Returns:
            List of CachedTrade objects, sorted by timestamp descending
        """
        address = address.lower()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, wallet_address, timestamp, price, size, fee, side, value, market_id, condition_id
                FROM trades
                WHERE wallet_address = ?
                ORDER BY timestamp DESC
            """, (address,))

            return [
                CachedTrade(
                    id=row[0],
                    wallet_address=row[1],
                    timestamp=row[2],
                    price=row[3],
                    size=row[4],
                    fee=row[5],
                    side=row[6],
                    value=row[7],
                    market_id=row[8] or "",
                    condition_id=row[9] or "",
                )
                for row in cursor.fetchall()
            ]

    def get_trades_since(self, address: str, since_timestamp: float) -> List[CachedTrade]:
        """
        Get trades newer than a specific timestamp.

        Args:
            address: Wallet address
            since_timestamp: Unix timestamp

        Returns:
            List of trades newer than since_timestamp
        """
        address = address.lower()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, wallet_address, timestamp, price, size, fee, side, value, market_id, condition_id
                FROM trades
                WHERE wallet_address = ? AND timestamp > ?
                ORDER BY timestamp DESC
            """, (address, since_timestamp))

            return [
                CachedTrade(
                    id=row[0],
                    wallet_address=row[1],
                    timestamp=row[2],
                    price=row[3],
                    size=row[4],
                    fee=row[5],
                    side=row[6],
                    value=row[7],
                    market_id=row[8] or "",
                    condition_id=row[9] or "",
                )
                for row in cursor.fetchall()
            ]

    def save_trades(self, address: str, trades: List[Any]) -> int:
        """
        Save trades to cache.

        Accepts either CachedTrade objects, GoldskyTrade objects, or dicts.

        Args:
            address: Wallet address
            trades: List of trade objects

        Returns:
            Number of trades saved (new insertions)
        """
        if not trades:
            return 0

        address = address.lower()
        now = datetime.utcnow().isoformat()
        saved_count = 0

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                for trade in trades:
                    try:
                        # Handle different trade formats
                        if isinstance(trade, CachedTrade):
                            trade_id = trade.id
                            timestamp = trade.timestamp
                            price = trade.price
                            size = trade.size
                            fee = trade.fee
                            side = trade.side
                            value = trade.value
                            market_id = trade.market_id
                            condition_id = trade.condition_id
                        elif isinstance(trade, dict):
                            trade_id = trade.get('id', f"{address}_{trade.get('timestamp', 0)}")
                            ts = trade.get('timestamp')
                            if hasattr(ts, 'timestamp'):
                                timestamp = ts.timestamp()
                            else:
                                timestamp = float(ts or 0)
                            price = float(trade.get('price', 0) or 0)
                            size = float(trade.get('size', 0) or 0)
                            fee = float(trade.get('fee', 0) or 0)
                            side = str(trade.get('side', 'BUY')).upper()
                            value = float(trade.get('value', 0) or 0) or (price * size)
                            market_id = trade.get('market_id', '') or ''
                            condition_id = trade.get('condition_id', '') or ''
                        else:
                            # GoldskyTrade or similar object
                            trade_id = getattr(trade, 'id', '') or f"{address}_{getattr(trade, 'timestamp', 0)}"
                            ts = getattr(trade, 'timestamp', None)
                            if hasattr(ts, 'timestamp'):
                                timestamp = ts.timestamp()
                            else:
                                timestamp = float(ts or 0)
                            price = float(getattr(trade, 'price', 0) or 0)
                            size = float(getattr(trade, 'size', 0) or 0)
                            fee = float(getattr(trade, 'fee', 0) or 0)
                            side = str(getattr(trade, 'side', 'BUY')).upper()
                            value = float(getattr(trade, 'value', 0) or 0) or (price * size)
                            market_id = getattr(trade, 'market_id', '') or ''
                            condition_id = getattr(trade, 'condition_id', '') or ''

                        # Insert or ignore (don't overwrite existing)
                        cursor = conn.execute("""
                            INSERT OR IGNORE INTO trades
                            (id, wallet_address, timestamp, price, size, fee, side, value, market_id, condition_id, fetched_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (trade_id, address, timestamp, price, size, fee, side, value, market_id, condition_id, now))

                        if cursor.rowcount > 0:
                            saved_count += 1

                    except Exception as e:
                        logger.debug(f"Error saving trade: {e}")

                # Update wallet sync info
                cursor = conn.execute("""
                    SELECT MIN(timestamp), MAX(timestamp), COUNT(*)
                    FROM trades WHERE wallet_address = ?
                """, (address,))
                row = cursor.fetchone()

                if row and row[2] > 0:
                    first_ts, last_ts, count = row
                    conn.execute("""
                        INSERT OR REPLACE INTO wallet_sync
                        (address, last_trade_timestamp, last_fetched_at, trade_count, first_trade_timestamp)
                        VALUES (?, ?, ?, ?, ?)
                    """, (address, last_ts, now, count, first_ts))

                conn.commit()

        return saved_count

    def append_trades(self, address: str, new_trades: List[Any]) -> int:
        """
        Append new trades to existing cache (incremental sync).

        Same as save_trades but with clearer intent.

        Args:
            address: Wallet address
            new_trades: New trades to append

        Returns:
            Number of trades added
        """
        return self.save_trades(address, new_trades)

    def has_trades(self, address: str) -> bool:
        """Check if wallet has any cached trades."""
        address = address.lower()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT 1 FROM trades WHERE wallet_address = ? LIMIT 1",
                (address,)
            )
            return cursor.fetchone() is not None

    def get_trade_count(self, address: str) -> int:
        """Get number of cached trades for a wallet."""
        address = address.lower()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM trades WHERE wallet_address = ?",
                (address,)
            )
            return cursor.fetchone()[0]

    # -------------------------------------------------------------------------
    # NumPy Conversion (for Numba processing)
    # -------------------------------------------------------------------------

    def get_trades_as_numpy(self, address: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Get trades as numpy arrays, ready for Numba processing.

        Returns dict with arrays:
            - prices: float64
            - sizes: float64
            - fees: float64
            - sides: int64 (1=BUY, 0=SELL)
            - timestamps: int64 (unix timestamps)
            - values: float64

        Args:
            address: Wallet address

        Returns:
            Dict of numpy arrays, or None if no trades
        """
        address = address.lower()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT price, size, fee, side, timestamp, value, market_id
                FROM trades
                WHERE wallet_address = ?
                ORDER BY timestamp ASC
            """, (address,))

            rows = cursor.fetchall()

        if not rows:
            return None

        n = len(rows)
        prices = np.zeros(n, dtype=np.float64)
        sizes = np.zeros(n, dtype=np.float64)
        fees = np.zeros(n, dtype=np.float64)
        sides = np.zeros(n, dtype=np.int64)
        timestamps = np.zeros(n, dtype=np.int64)
        values = np.zeros(n, dtype=np.float64)
        market_ids = []

        for i, row in enumerate(rows):
            prices[i] = row[0]
            sizes[i] = row[1]
            fees[i] = row[2]
            sides[i] = 1 if row[3] == 'BUY' else 0
            timestamps[i] = int(row[4])
            values[i] = row[5]
            market_ids.append(row[6] or "")

        # Convert market_ids to integer indices
        unique_markets = list(set(market_ids))
        market_to_idx = {m: i for i, m in enumerate(unique_markets)}
        market_id_arr = np.array([market_to_idx[m] for m in market_ids], dtype=np.int64)

        return {
            'prices': prices,
            'sizes': sizes,
            'fees': fees,
            'sides': sides,
            'timestamps': timestamps,
            'values': values,
            'market_ids': market_id_arr,
            'n_markets': len(unique_markets),
        }

    def get_trades_as_dicts(self, address: str) -> List[Dict[str, Any]]:
        """
        Get trades as list of dicts (for FastWalletData.from_trades).

        Args:
            address: Wallet address

        Returns:
            List of trade dicts compatible with FastWalletData.from_trades
        """
        address = address.lower()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT price, size, fee, side, timestamp, value, market_id, condition_id
                FROM trades
                WHERE wallet_address = ?
                ORDER BY timestamp ASC
            """, (address,))

            return [
                {
                    'price': row[0],
                    'size': row[1],
                    'fee': row[2],
                    'side': row[3],
                    'timestamp': row[4],
                    'value': row[5],
                    'market_id': row[6] or row[7] or "",
                    'condition_id': row[7] or "",
                }
                for row in cursor.fetchall()
            ]

    # -------------------------------------------------------------------------
    # Batch Operations
    # -------------------------------------------------------------------------

    def get_batch_trades(self, addresses: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get trades for multiple wallets in a single query.

        Args:
            addresses: List of wallet addresses

        Returns:
            Dict mapping address -> list of trade dicts
        """
        if not addresses:
            return {}

        addresses = [a.lower() for a in addresses]

        result = {addr: [] for addr in addresses}

        with sqlite3.connect(self.db_path) as conn:
            # Use a single query with IN clause for efficiency
            placeholders = ','.join('?' * len(addresses))
            cursor = conn.execute(f"""
                SELECT wallet_address, price, size, fee, side, timestamp, value, market_id, condition_id
                FROM trades
                WHERE wallet_address IN ({placeholders})
                ORDER BY wallet_address, timestamp ASC
            """, addresses)

            for row in cursor.fetchall():
                wallet = row[0]
                result[wallet].append({
                    'price': row[1],
                    'size': row[2],
                    'fee': row[3],
                    'side': row[4],
                    'timestamp': row[5],
                    'value': row[6],
                    'market_id': row[7] or row[8] or "",
                    'condition_id': row[8] or "",
                })

        return result

    def get_cached_wallets(self) -> List[str]:
        """Get all wallet addresses that have cached trades."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT address FROM wallet_sync")
            return [row[0] for row in cursor.fetchall()]

    # -------------------------------------------------------------------------
    # Sync Status Operations
    # -------------------------------------------------------------------------

    def get_last_timestamp(self, address: str) -> float:
        """
        Get the timestamp of the most recent cached trade for a wallet.

        Used for incremental sync - fetch only trades newer than this.

        Args:
            address: Wallet address

        Returns:
            Unix timestamp, or 0 if no cached trades
        """
        address = address.lower()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT last_trade_timestamp FROM wallet_sync WHERE address = ?",
                (address,)
            )
            row = cursor.fetchone()
            return row[0] if row else 0.0

    def get_sync_info(self, address: str) -> Optional[WalletSyncInfo]:
        """Get sync status for a wallet."""
        address = address.lower()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT address, last_trade_timestamp, last_fetched_at, trade_count
                FROM wallet_sync WHERE address = ?
            """, (address,))
            row = cursor.fetchone()
            if row:
                return WalletSyncInfo(
                    address=row[0],
                    last_trade_timestamp=row[1],
                    last_fetched_at=datetime.fromisoformat(row[2]),
                    trade_count=row[3],
                )
        return None

    def get_stale_wallets(self, max_age_hours: int = 24) -> List[str]:
        """
        Get wallets that need a refresh (haven't been synced recently).

        Args:
            max_age_hours: Consider stale if not synced within this many hours

        Returns:
            List of wallet addresses that need refresh
        """
        cutoff = (datetime.utcnow() - timedelta(hours=max_age_hours)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT address FROM wallet_sync
                WHERE last_fetched_at < ?
            """, (cutoff,))
            return [row[0] for row in cursor.fetchall()]

    def get_uncached_wallets(self, addresses: List[str]) -> List[str]:
        """
        Get addresses that are NOT in the cache.

        Args:
            addresses: List of wallet addresses to check

        Returns:
            List of addresses with no cached data
        """
        if not addresses:
            return []

        addresses = [a.lower() for a in addresses]
        cached = set(self.get_cached_wallets())

        return [addr for addr in addresses if addr not in cached]

    # -------------------------------------------------------------------------
    # Cache Management
    # -------------------------------------------------------------------------

    def invalidate(self, address: str):
        """
        Invalidate cache for a specific wallet (force refresh).

        Args:
            address: Wallet address
        """
        address = address.lower()
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM trades WHERE wallet_address = ?", (address,))
                conn.execute("DELETE FROM wallet_sync WHERE address = ?", (address,))
                conn.commit()

    def prune(self, max_age_days: int = 90):
        """
        Remove old trades from cache.

        Args:
            max_age_days: Remove trades older than this
        """
        cutoff_ts = (datetime.utcnow() - timedelta(days=max_age_days)).timestamp()

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                # Delete old trades
                cursor = conn.execute(
                    "DELETE FROM trades WHERE timestamp < ?",
                    (cutoff_ts,)
                )
                deleted = cursor.rowcount

                # Update sync info for affected wallets
                conn.execute("""
                    UPDATE wallet_sync SET
                        trade_count = (
                            SELECT COUNT(*) FROM trades WHERE trades.wallet_address = wallet_sync.address
                        ),
                        last_trade_timestamp = COALESCE(
                            (SELECT MAX(timestamp) FROM trades WHERE trades.wallet_address = wallet_sync.address),
                            0
                        ),
                        first_trade_timestamp = COALESCE(
                            (SELECT MIN(timestamp) FROM trades WHERE trades.wallet_address = wallet_sync.address),
                            0
                        )
                """)

                # Remove wallets with no trades left
                conn.execute("DELETE FROM wallet_sync WHERE trade_count = 0")

                conn.commit()

        logger.info(f"Pruned {deleted} trades older than {max_age_days} days")

    def clear(self):
        """Clear all cached data (full rebuild required)."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM trades")
                conn.execute("DELETE FROM wallet_sync")
                conn.commit()
        logger.info("Trade cache cleared")

    def vacuum(self):
        """Reclaim disk space after deletions."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("VACUUM")

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def stats(self) -> TradeCacheStats:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            # Count totals
            wallets = conn.execute("SELECT COUNT(*) FROM wallet_sync").fetchone()[0]
            trades = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]

            # Count stale (24h)
            cutoff = (datetime.utcnow() - timedelta(hours=24)).isoformat()
            stale = conn.execute(
                "SELECT COUNT(*) FROM wallet_sync WHERE last_fetched_at < ?",
                (cutoff,)
            ).fetchone()[0]

            # Get sync time range
            sync_range = conn.execute("""
                SELECT MIN(last_fetched_at), MAX(last_fetched_at) FROM wallet_sync
            """).fetchone()

            oldest = datetime.fromisoformat(sync_range[0]) if sync_range[0] else None
            newest = datetime.fromisoformat(sync_range[1]) if sync_range[1] else None

        # Get file size
        size_mb = self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0

        return TradeCacheStats(
            total_wallets=wallets,
            total_trades=trades,
            stale_wallets=stale,
            cache_size_mb=size_mb,
            oldest_sync=oldest,
            newest_sync=newest,
        )

    def __repr__(self) -> str:
        stats = self.stats()
        return (
            f"TradeCache(wallets={stats.total_wallets:,}, "
            f"trades={stats.total_trades:,}, "
            f"size={stats.cache_size_mb:.1f}MB)"
        )


# Global cache instance
_trade_cache: Optional[TradeCache] = None


def get_trade_cache(db_path: str = "data/trade_cache.db") -> TradeCache:
    """Get or create global trade cache instance."""
    global _trade_cache
    if _trade_cache is None:
        _trade_cache = TradeCache(db_path=db_path)
    return _trade_cache
