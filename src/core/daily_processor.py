"""
Daily Wallet Processor - Continuous collection + Daily batch ranking.

Architecture:
    1. Continuous Scanner: Watches markets, collects new wallet addresses
    2. Master Wallet List: Stores every wallet ever seen (deduplicated)
    3. Daily Processor: Calculates 117 metrics for all wallets every 24h
    4. Rankings: Top 100 per metric for 7d, 1m, all-time

Storage:
    - PostgreSQL: Master wallet list + metrics + rankings
    - Redis: Hot cache for fast queries
"""
import asyncio
import time
import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set, Tuple, Literal
from dataclasses import dataclass, field
from pathlib import Path
import sqlite3
import threading

logger = logging.getLogger(__name__)

# Time periods for rankings
TimePeriod = Literal["7d", "31d", "all_time"]
TIME_PERIODS: List[TimePeriod] = ["7d", "31d", "all_time"]


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class WalletRecord:
    """A wallet in the master list."""
    address: str
    first_seen: datetime
    last_seen: datetime
    total_trades: int = 0
    last_processed: Optional[datetime] = None


@dataclass
class MetricRanking:
    """Ranking entry for a single metric."""
    metric_id: int
    time_period: TimePeriod
    rank: int
    address: str
    value: float
    updated_at: datetime


@dataclass
class DailyProcessorStats:
    """Stats from a daily processing run."""
    total_wallets: int = 0
    wallets_processed: int = 0
    wallets_failed: int = 0
    new_wallets_added: int = 0

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def duration_seconds(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


# =============================================================================
# WALLET STORAGE (SQLite for simplicity, can swap to PostgreSQL)
# =============================================================================

class WalletStorage:
    """
    Persistent storage for wallets, metrics, and rankings.

    Uses SQLite by default (simple, no external deps).
    Can be swapped to PostgreSQL for production.
    """

    def __init__(self, db_path: str = "data/polyscan.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._lock = threading.Lock()

    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                -- Master wallet list
                CREATE TABLE IF NOT EXISTS wallets (
                    address TEXT PRIMARY KEY,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    total_trades INTEGER DEFAULT 0,
                    last_processed TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_wallets_last_seen ON wallets(last_seen);

                -- Wallet metrics (latest values)
                CREATE TABLE IF NOT EXISTS wallet_metrics (
                    address TEXT NOT NULL,
                    time_period TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    quant_score REAL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (address, time_period)
                );

                -- Rankings (top 100 per metric per time period)
                CREATE TABLE IF NOT EXISTS rankings (
                    metric_id INTEGER NOT NULL,
                    time_period TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    address TEXT NOT NULL,
                    value REAL NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (metric_id, time_period, rank)
                );
                CREATE INDEX IF NOT EXISTS idx_rankings_address ON rankings(address);

                -- Processing history
                CREATE TABLE IF NOT EXISTS processing_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    total_wallets INTEGER,
                    wallets_processed INTEGER,
                    wallets_failed INTEGER,
                    status TEXT DEFAULT 'running'
                );
            """)

    # -------------------------------------------------------------------------
    # Wallet operations
    # -------------------------------------------------------------------------

    def add_wallets(self, addresses: List[str]) -> int:
        """
        Add wallets to master list (deduplicated).

        Returns number of NEW wallets added.
        """
        now = datetime.utcnow().isoformat()
        new_count = 0

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                for addr in addresses:
                    addr = addr.lower()
                    cursor = conn.execute(
                        "SELECT address FROM wallets WHERE address = ?",
                        (addr,)
                    )
                    if cursor.fetchone() is None:
                        conn.execute(
                            "INSERT INTO wallets (address, first_seen, last_seen) VALUES (?, ?, ?)",
                            (addr, now, now)
                        )
                        new_count += 1
                    else:
                        conn.execute(
                            "UPDATE wallets SET last_seen = ? WHERE address = ?",
                            (now, addr)
                        )
                conn.commit()

        return new_count

    def get_all_wallets(self) -> List[str]:
        """Get all wallet addresses from master list."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT address FROM wallets")
            return [row[0] for row in cursor.fetchall()]

    def get_wallet_count(self) -> int:
        """Get total number of wallets in master list."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM wallets")
            return cursor.fetchone()[0]

    def get_wallets_since(self, since: datetime) -> List[str]:
        """Get wallets seen since a specific time."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT address FROM wallets WHERE last_seen >= ?",
                (since.isoformat(),)
            )
            return [row[0] for row in cursor.fetchall()]

    # -------------------------------------------------------------------------
    # Metrics operations
    # -------------------------------------------------------------------------

    def save_wallet_metrics(
        self,
        address: str,
        time_period: TimePeriod,
        metrics: Dict[int, float],
        quant_score: float = 0.0
    ):
        """Save metrics for a wallet."""
        now = datetime.utcnow().isoformat()
        metrics_json = json.dumps({str(k): v for k, v in metrics.items()})

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO wallet_metrics
                    (address, time_period, metrics_json, quant_score, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (address.lower(), time_period, metrics_json, quant_score, now))

                # Update last_processed
                conn.execute(
                    "UPDATE wallets SET last_processed = ? WHERE address = ?",
                    (now, address.lower())
                )
                conn.commit()

    def get_wallet_metrics(
        self,
        address: str,
        time_period: TimePeriod
    ) -> Optional[Dict[int, float]]:
        """Get metrics for a wallet."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT metrics_json FROM wallet_metrics WHERE address = ? AND time_period = ?",
                (address.lower(), time_period)
            )
            row = cursor.fetchone()
            if row:
                data = json.loads(row[0])
                return {int(k): v for k, v in data.items()}
        return None

    def get_all_metrics_for_period(
        self,
        time_period: TimePeriod
    ) -> Dict[str, Dict[int, float]]:
        """Get all wallet metrics for a time period (for ranking)."""
        result = {}
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT address, metrics_json FROM wallet_metrics WHERE time_period = ?",
                (time_period,)
            )
            for address, metrics_json in cursor.fetchall():
                data = json.loads(metrics_json)
                result[address] = {int(k): v for k, v in data.items()}
        return result

    # -------------------------------------------------------------------------
    # Rankings operations
    # -------------------------------------------------------------------------

    def update_rankings(self, time_period: TimePeriod, all_metrics: Dict[str, Dict[int, float]]):
        """
        Update top 100 rankings for all 117 metrics.

        Args:
            time_period: "7d", "31d", or "all_time"
            all_metrics: Dict mapping address -> {metric_id: value}
        """
        now = datetime.utcnow().isoformat()

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                # Clear old rankings for this period
                conn.execute(
                    "DELETE FROM rankings WHERE time_period = ?",
                    (time_period,)
                )

                # Calculate rankings for each metric
                for metric_id in range(1, 118):
                    # Get all values for this metric (skip None/NaN values)
                    values = []
                    for address, metrics in all_metrics.items():
                        if metric_id in metrics:
                            val = metrics[metric_id]
                            if val is not None and val == val:  # Check not None and not NaN
                                values.append((address, float(val)))

                    # Sort by value (descending for most metrics)
                    # Note: Some metrics like drawdown are "lower is better"
                    # but we'll rank by raw value for simplicity
                    values.sort(key=lambda x: x[1], reverse=True)

                    # Insert top 100
                    for rank, (address, value) in enumerate(values[:100], start=1):
                        conn.execute("""
                            INSERT INTO rankings
                            (metric_id, time_period, rank, address, value, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (metric_id, time_period, rank, address, value, now))

                conn.commit()

        logger.info(f"Updated rankings for {time_period}: 117 metrics × top 100")

    def get_rankings(
        self,
        metric_id: int,
        time_period: TimePeriod,
        limit: int = 100
    ) -> List[Tuple[int, str, float]]:
        """
        Get rankings for a specific metric.

        Returns list of (rank, address, value) tuples.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT rank, address, value FROM rankings
                WHERE metric_id = ? AND time_period = ?
                ORDER BY rank
                LIMIT ?
            """, (metric_id, time_period, limit))
            return cursor.fetchall()

    def get_wallet_rankings(self, address: str) -> Dict[TimePeriod, Dict[int, int]]:
        """
        Get all rankings for a specific wallet.

        Returns: {time_period: {metric_id: rank}}
        """
        result = {period: {} for period in TIME_PERIODS}

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT time_period, metric_id, rank FROM rankings
                WHERE address = ?
            """, (address.lower(),))
            for time_period, metric_id, rank in cursor.fetchall():
                result[time_period][metric_id] = rank

        return result

    # -------------------------------------------------------------------------
    # Processing history
    # -------------------------------------------------------------------------

    def start_processing_run(self, total_wallets: int) -> int:
        """Start a new processing run, returns run ID."""
        now = datetime.utcnow().isoformat()
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT INTO processing_runs (started_at, total_wallets, status)
                    VALUES (?, ?, 'running')
                """, (now, total_wallets))
                conn.commit()
                return cursor.lastrowid

    def complete_processing_run(
        self,
        run_id: int,
        wallets_processed: int,
        wallets_failed: int
    ):
        """Mark a processing run as complete."""
        now = datetime.utcnow().isoformat()
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE processing_runs
                    SET completed_at = ?, wallets_processed = ?, wallets_failed = ?, status = 'completed'
                    WHERE id = ?
                """, (now, wallets_processed, wallets_failed, run_id))
                conn.commit()


# =============================================================================
# CONTINUOUS WALLET COLLECTOR
# =============================================================================

class WalletCollector:
    """
    Continuously scans markets and collects wallet addresses.

    Runs in background, adding new wallets to the master list.
    """

    def __init__(
        self,
        storage: WalletStorage,
        scan_interval_minutes: int = 5,
    ):
        self.storage = storage
        self.scan_interval = scan_interval_minutes * 60
        self._running = False
        self._task = None

    async def _fetch_active_wallets(self) -> List[str]:
        """
        Fetch wallets that have recent activity across all markets.

        Uses Goldsky orderbook subgraph to get recent trades and extract
        unique wallet addresses (both makers and takers).

        OPTIMIZED: Fetches multiple pages in PARALLEL for speed.
        """
        import aiohttp
        from src.api.goldsky_client import GOLDSKY_ENDPOINTS, ALL_RECENT_TRADES_QUERY

        try:
            wallets: Set[str] = set()

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60),
                connector=aiohttp.TCPConnector(limit=50),
            ) as session:

                # Fetch MANY pages in PARALLEL
                first = 1000
                max_pages = 50  # 50,000 trades = ~25,000 unique wallets

                async def fetch_page(skip: int) -> List[str]:
                    """Fetch a single page of trades and extract wallets."""
                    page_wallets = []
                    try:
                        async with session.post(
                            GOLDSKY_ENDPOINTS["orderbook"],
                            json={
                                "query": ALL_RECENT_TRADES_QUERY,
                                "variables": {"first": first, "skip": skip},
                            },
                            headers={"Content-Type": "application/json"},
                        ) as response:
                            if response.status != 200:
                                return []

                            result = await response.json()
                            if "errors" in result:
                                return []

                            data = result.get("data", {})
                            events = data.get("orderFilledEvents", [])

                            for event in events:
                                maker = event.get("maker", "")
                                taker = event.get("taker", "")
                                if maker:
                                    page_wallets.append(maker.lower())
                                if taker:
                                    page_wallets.append(taker.lower())

                    except Exception as e:
                        logger.debug(f"Error fetching page at skip={skip}: {e}")

                    return page_wallets

                # Create tasks for all pages
                tasks = [fetch_page(skip) for skip in range(0, first * max_pages, first)]

                # Run all in parallel
                results = await asyncio.gather(*tasks)

                # Collect all wallets
                for page_wallets in results:
                    wallets.update(page_wallets)

            logger.info(f"Fetched {len(wallets)} unique active wallets from {max_pages} pages")
            return list(wallets)

        except Exception as e:
            logger.error(f"Error fetching active wallets: {e}")
            return []

    async def _collect_loop(self):
        """Main collection loop."""
        logger.info(f"Wallet collector started (interval: {self.scan_interval}s)")

        while self._running:
            try:
                # Fetch active wallets
                wallets = await self._fetch_active_wallets()

                if wallets:
                    new_count = self.storage.add_wallets(wallets)
                    total = self.storage.get_wallet_count()
                    logger.info(f"Collected {len(wallets)} wallets, {new_count} new. Total: {total:,}")

            except Exception as e:
                logger.error(f"Collection error: {e}")

            # Wait for next interval
            await asyncio.sleep(self.scan_interval)

    def start(self):
        """Start the collector in background."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._collect_loop())
        logger.info("Wallet collector started")

    def stop(self):
        """Stop the collector."""
        self._running = False
        if self._task:
            self._task.cancel()
        logger.info("Wallet collector stopped")

    def add_wallets_manually(self, addresses: List[str]) -> int:
        """Manually add wallets to the master list."""
        return self.storage.add_wallets(addresses)


# =============================================================================
# DAILY BATCH PROCESSOR
# =============================================================================

class DailyProcessor:
    """
    Processes all wallets daily, calculates metrics, and updates rankings.

    Run once per day (e.g., via cron at midnight).

    With TradeCache:
        - First run: Fetch from API, save to cache (30 wallets/sec)
        - Subsequent runs: Load from cache (50,000+ wallets/sec)
        - Incremental: Only fetch new trades for stale wallets
    """

    def __init__(
        self,
        storage: WalletStorage,
        chunk_size: int = 1000,
        max_concurrent_http: int = 200,
        use_trade_cache: bool = True,
        trade_cache_path: str = "data/trade_cache.db",
        cache_max_age_hours: int = 24,
    ):
        self.storage = storage
        self.chunk_size = chunk_size
        self.max_concurrent_http = max_concurrent_http
        self.use_trade_cache = use_trade_cache
        self.cache_max_age_hours = cache_max_age_hours

        # Initialize trade cache
        self._trade_cache = None
        if use_trade_cache:
            from src.cache.trade_cache import TradeCache
            self._trade_cache = TradeCache(db_path=trade_cache_path)

    async def process_all_wallets(
        self,
        progress_callback: Optional[callable] = None,
        force_refresh: bool = False,
    ) -> DailyProcessorStats:
        """
        Process all wallets in the master list with MAXIMUM parallelism.

        With TradeCache enabled:
            1. Load trades from cache (50,000+ wallets/sec)
            2. Fetch from API only for uncached/stale wallets
            3. Save new trades to cache for future runs
            4. Calculate 117 metrics for 7d, 31d, all_time
            5. Update rankings

        Args:
            progress_callback: Called with (processed, total, stage)
            force_refresh: If True, fetch from API even if cached

        Returns:
            DailyProcessorStats with processing summary
        """
        import gc
        import numpy as np
        from src.api.goldsky_client import GoldskyClient
        from src.core.analyzer.numba_metrics import (
            FastWalletData, calculate_117_metrics, calculate_quant_score
        )

        stats = DailyProcessorStats()
        stats.start_time = datetime.utcnow()

        # Get all wallets
        all_wallets = self.storage.get_all_wallets()
        stats.total_wallets = len(all_wallets)

        if stats.total_wallets == 0:
            logger.warning("No wallets to process")
            stats.end_time = datetime.utcnow()
            return stats

        # Start processing run
        run_id = self.storage.start_processing_run(stats.total_wallets)
        logger.info(f"Starting daily processing of {stats.total_wallets:,} wallets")

        # Collect all metrics for ranking at the end
        all_metrics_by_period: Dict[TimePeriod, Dict[str, Dict[int, float]]] = {
            "7d": {},
            "31d": {},
            "all_time": {},
        }

        # Time cutoffs for period filtering
        now = datetime.utcnow()
        cutoff_7d = (now - timedelta(days=7)).timestamp()
        cutoff_31d = (now - timedelta(days=31)).timestamp()

        # =====================================================================
        # CACHE-ENABLED PROCESSING
        # =====================================================================
        if self._trade_cache and not force_refresh:
            logger.info("Trade cache enabled - checking for cached data...")
            cache_stats = self._trade_cache.stats()
            logger.info(f"Cache stats: {cache_stats.total_wallets:,} wallets, {cache_stats.total_trades:,} trades")

            # Determine which wallets need API fetch
            uncached_wallets = self._trade_cache.get_uncached_wallets(all_wallets)
            stale_wallets = set(self._trade_cache.get_stale_wallets(max_age_hours=self.cache_max_age_hours))
            wallets_needing_fetch = set(uncached_wallets) | stale_wallets
            cached_wallets = [w for w in all_wallets if w.lower() not in wallets_needing_fetch]

            logger.info(
                f"Cache status: {len(cached_wallets):,} cached, "
                f"{len(uncached_wallets):,} uncached, {len(stale_wallets):,} stale"
            )

            # -----------------------------------------------------------------
            # PHASE 1: Process cached wallets at 50,000+ wallets/sec
            # -----------------------------------------------------------------
            if cached_wallets:
                if progress_callback:
                    progress_callback(0, stats.total_wallets, "loading_cache")

                logger.info(f"Processing {len(cached_wallets):,} wallets from cache...")
                cache_start = time.perf_counter()

                # Load trades in batches from cache
                CACHE_BATCH_SIZE = 1000
                for batch_idx in range(0, len(cached_wallets), CACHE_BATCH_SIZE):
                    batch = cached_wallets[batch_idx:batch_idx + CACHE_BATCH_SIZE]
                    batch_trades = self._trade_cache.get_batch_trades(batch)

                    for address in batch:
                        trades_data = batch_trades.get(address.lower(), [])

                        if not trades_data:
                            stats.wallets_failed += 1
                            continue

                        try:
                            # Filter trades by time period
                            trades_7d = [t for t in trades_data if t['timestamp'] >= cutoff_7d]
                            trades_31d = [t for t in trades_data if t['timestamp'] >= cutoff_31d]

                            # Calculate metrics for each period
                            for period, trades in [
                                ("7d", trades_7d),
                                ("31d", trades_31d),
                                ("all_time", trades_data),
                            ]:
                                if not trades or len(trades) < 2:
                                    continue

                                # Calculate metrics
                                wallet_data = FastWalletData.from_trades(address, trades)
                                metrics = calculate_117_metrics(wallet_data)
                                quant_score = calculate_quant_score(metrics)

                                # Save metrics
                                self.storage.save_wallet_metrics(address, period, metrics, quant_score)

                                # Collect for ranking
                                all_metrics_by_period[period][address] = metrics

                            stats.wallets_processed += 1

                        except Exception as e:
                            logger.debug(f"Error processing cached {address}: {e}")
                            stats.wallets_failed += 1

                    # Progress callback
                    if progress_callback:
                        progress_callback(
                            stats.wallets_processed + stats.wallets_failed,
                            stats.total_wallets,
                            "cache_processing"
                        )

                cache_elapsed = time.perf_counter() - cache_start
                cache_rate = len(cached_wallets) / cache_elapsed if cache_elapsed > 0 else 0
                logger.info(
                    f"Cache processing complete: {len(cached_wallets):,} wallets in "
                    f"{cache_elapsed:.1f}s ({cache_rate:,.0f} wallets/sec)"
                )

            # -----------------------------------------------------------------
            # PHASE 2: Fetch and process uncached/stale wallets from API
            # -----------------------------------------------------------------
            wallets_to_fetch = list(wallets_needing_fetch)
            if wallets_to_fetch:
                logger.info(f"Fetching {len(wallets_to_fetch):,} wallets from API...")
                await self._fetch_and_process_wallets(
                    wallets_to_fetch,
                    stats,
                    all_metrics_by_period,
                    cutoff_7d,
                    cutoff_31d,
                    progress_callback,
                    save_to_cache=True,
                )

        # =====================================================================
        # ORIGINAL PROCESSING (no cache or force refresh)
        # =====================================================================
        else:
            if force_refresh:
                logger.info("Force refresh enabled - fetching all from API")
            elif not self._trade_cache:
                logger.info("Trade cache disabled - fetching all from API")

            await self._fetch_and_process_wallets(
                all_wallets,
                stats,
                all_metrics_by_period,
                cutoff_7d,
                cutoff_31d,
                progress_callback,
                save_to_cache=self._trade_cache is not None,
            )

        # Update rankings for all periods
        logger.info("Updating rankings...")
        for period in TIME_PERIODS:
            if all_metrics_by_period[period]:
                self.storage.update_rankings(period, all_metrics_by_period[period])

        if progress_callback:
            progress_callback(stats.total_wallets, stats.total_wallets, "rankings")

        # Complete processing run
        stats.end_time = datetime.utcnow()
        self.storage.complete_processing_run(run_id, stats.wallets_processed, stats.wallets_failed)

        logger.info(
            f"Daily processing complete: {stats.wallets_processed:,} processed, "
            f"{stats.wallets_failed:,} failed in {stats.duration_seconds:.1f}s"
        )

        # Log cache stats if enabled
        if self._trade_cache:
            cache_stats = self._trade_cache.stats()
            logger.info(f"Final cache: {cache_stats.total_wallets:,} wallets, {cache_stats.total_trades:,} trades")

        return stats

    async def _fetch_and_process_wallets(
        self,
        wallets: List[str],
        stats: DailyProcessorStats,
        all_metrics_by_period: Dict[TimePeriod, Dict[str, Dict[int, float]]],
        cutoff_7d: float,
        cutoff_31d: float,
        progress_callback: Optional[callable],
        save_to_cache: bool = False,
    ):
        """
        Fetch trades from API and process wallets.

        Internal helper method used by process_all_wallets.
        """
        import gc
        from src.api.goldsky_client import GoldskyClient
        from src.core.analyzer.numba_metrics import (
            FastWalletData, calculate_117_metrics, calculate_quant_score
        )

        if not wallets:
            return

        # Process in PARALLEL super-batches (10 batches of 100 = 1000 wallets at once)
        PARALLEL_BATCHES = 10
        BATCH_SIZE = 100
        SUPER_BATCH_SIZE = PARALLEL_BATCHES * BATCH_SIZE  # 1000 wallets

        total_super_batches = (len(wallets) + SUPER_BATCH_SIZE - 1) // SUPER_BATCH_SIZE

        for super_idx in range(total_super_batches):
            super_start = super_idx * SUPER_BATCH_SIZE
            super_end = min(super_start + SUPER_BATCH_SIZE, len(wallets))
            super_batch_wallets = wallets[super_start:super_end]

            logger.info(f"API batch {super_idx + 1}/{total_super_batches} ({len(super_batch_wallets)} wallets)")

            # Split into parallel batches
            batches = [
                super_batch_wallets[i:i + BATCH_SIZE]
                for i in range(0, len(super_batch_wallets), BATCH_SIZE)
            ]

            # Fetch ALL batches in parallel
            async def fetch_batch(batch_wallets):
                client = GoldskyClient(batch_size=100, requests_per_second=50.0)
                try:
                    return await client.get_batch_trades(batch_wallets, max_per_wallet=500)
                except Exception as e:
                    logger.debug(f"Batch fetch error: {e}")
                    return {}

            # Run all batch fetches in parallel
            batch_results = await asyncio.gather(*[fetch_batch(b) for b in batches])

            # Merge results
            all_trades_by_wallet = {}
            for result in batch_results:
                all_trades_by_wallet.update(result)

            # Process all wallets from this super-batch
            for address in super_batch_wallets:
                all_trades = all_trades_by_wallet.get(address.lower(), [])

                if not all_trades:
                    stats.wallets_failed += 1
                    continue

                try:
                    # Save to cache if enabled
                    if save_to_cache and self._trade_cache:
                        self._trade_cache.save_trades(address, all_trades)

                    # Filter trades by time period
                    trades_7d = [t for t in all_trades if t.timestamp.timestamp() >= cutoff_7d]
                    trades_31d = [t for t in all_trades if t.timestamp.timestamp() >= cutoff_31d]

                    # Calculate metrics for each period
                    for period, trades in [
                        ("7d", trades_7d),
                        ("31d", trades_31d),
                        ("all_time", all_trades),
                    ]:
                        if not trades or len(trades) < 2:
                            continue

                        # Convert GoldskyTrade objects to dict format
                        trades_data = [
                            {
                                'price': t.price,
                                'size': t.size,
                                'fee': t.fee,
                                'side': t.side,
                                'timestamp': t.timestamp.timestamp(),
                                'market_id': t.condition_id or t.market_id,
                                'value': t.value,
                            }
                            for t in trades
                        ]

                        # Calculate metrics
                        wallet_data = FastWalletData.from_trades(address, trades_data)
                        metrics = calculate_117_metrics(wallet_data)
                        quant_score = calculate_quant_score(metrics)

                        # Save metrics
                        self.storage.save_wallet_metrics(address, period, metrics, quant_score)

                        # Collect for ranking
                        all_metrics_by_period[period][address] = metrics

                    stats.wallets_processed += 1

                except Exception as e:
                    logger.debug(f"Error processing {address}: {e}")
                    stats.wallets_failed += 1

            # Progress callback
            if progress_callback:
                progress_callback(
                    stats.wallets_processed + stats.wallets_failed,
                    stats.total_wallets,
                    "api_processing"
                )

            # Clear memory
            del all_trades_by_wallet
            gc.collect()

    def process_all_wallets_sync(
        self,
        progress_callback: Optional[callable] = None,
    ) -> DailyProcessorStats:
        """Synchronous wrapper for process_all_wallets."""
        return asyncio.run(self.process_all_wallets(progress_callback))


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

class PolyscanBot:
    """
    Main bot that orchestrates wallet collection and daily processing.

    Usage:
        bot = PolyscanBot()

        # Add wallets manually
        bot.add_wallets(["0x123...", "0x456..."])

        # Run daily processing (uses cache for 50,000+ wallets/sec)
        bot.run_daily_processing()

        # Get rankings
        top_sharpe = bot.get_top_100(metric_id=11, period="all_time")

        # Cache management
        print(bot.get_cache_stats())
        bot.clear_cache()  # Full rebuild
    """

    def __init__(
        self,
        db_path: str = "data/polyscan.db",
        use_trade_cache: bool = True,
        trade_cache_path: str = "data/trade_cache.db",
    ):
        self.storage = WalletStorage(db_path)
        self.collector = WalletCollector(self.storage)
        self.processor = DailyProcessor(
            self.storage,
            use_trade_cache=use_trade_cache,
            trade_cache_path=trade_cache_path,
        )

    # -------------------------------------------------------------------------
    # Wallet management
    # -------------------------------------------------------------------------

    def add_wallets(self, addresses: List[str]) -> int:
        """Add wallets to master list. Returns count of new wallets."""
        return self.storage.add_wallets(addresses)

    def get_wallet_count(self) -> int:
        """Get total wallets in master list."""
        return self.storage.get_wallet_count()

    def get_all_wallets(self) -> List[str]:
        """Get all wallet addresses."""
        return self.storage.get_all_wallets()

    # -------------------------------------------------------------------------
    # Background collection
    # -------------------------------------------------------------------------

    def start_collector(self):
        """Start background wallet collection."""
        self.collector.start()

    def stop_collector(self):
        """Stop background wallet collection."""
        self.collector.stop()

    # -------------------------------------------------------------------------
    # Daily processing
    # -------------------------------------------------------------------------

    def run_daily_processing(
        self,
        progress_callback: Optional[callable] = None
    ) -> DailyProcessorStats:
        """
        Run daily processing: calculate metrics + update rankings.

        Call this once per day (e.g., via cron).
        """
        return self.processor.process_all_wallets_sync(progress_callback)

    # -------------------------------------------------------------------------
    # Rankings & queries
    # -------------------------------------------------------------------------

    def get_top_100(
        self,
        metric_id: int,
        period: TimePeriod = "all_time",
    ) -> List[Tuple[int, str, float]]:
        """
        Get top 100 wallets for a specific metric.

        Args:
            metric_id: Metric ID (1-117)
            period: "7d", "31d", or "all_time"

        Returns:
            List of (rank, address, value) tuples
        """
        return self.storage.get_rankings(metric_id, period)

    def get_wallet_metrics(
        self,
        address: str,
        period: TimePeriod = "all_time",
    ) -> Optional[Dict[int, float]]:
        """Get all 117 metrics for a wallet."""
        return self.storage.get_wallet_metrics(address, period)

    def get_wallet_rankings(self, address: str) -> Dict[TimePeriod, Dict[int, int]]:
        """Get all rankings for a specific wallet."""
        return self.storage.get_wallet_rankings(address)

    # -------------------------------------------------------------------------
    # Cache management
    # -------------------------------------------------------------------------

    def get_cache_stats(self) -> Optional[Dict]:
        """
        Get trade cache statistics.

        Returns:
            Dict with cache stats, or None if cache disabled
        """
        if self.processor._trade_cache:
            stats = self.processor._trade_cache.stats()
            return {
                "total_wallets": stats.total_wallets,
                "total_trades": stats.total_trades,
                "stale_wallets": stats.stale_wallets,
                "cache_size_mb": stats.cache_size_mb,
                "oldest_sync": stats.oldest_sync.isoformat() if stats.oldest_sync else None,
                "newest_sync": stats.newest_sync.isoformat() if stats.newest_sync else None,
            }
        return None

    def clear_cache(self):
        """Clear all cached trade data (full rebuild required)."""
        if self.processor._trade_cache:
            self.processor._trade_cache.clear()
            logger.info("Trade cache cleared")

    def prune_cache(self, max_age_days: int = 90):
        """Remove old trades from cache."""
        if self.processor._trade_cache:
            self.processor._trade_cache.prune(max_age_days=max_age_days)

    def invalidate_wallet_cache(self, address: str):
        """Force refresh for a specific wallet."""
        if self.processor._trade_cache:
            self.processor._trade_cache.invalidate(address)

    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------

    def get_stats(self) -> Dict:
        """Get overall stats including cache."""
        stats = {
            "total_wallets": self.storage.get_wallet_count(),
            "db_path": str(self.storage.db_path),
        }

        # Add cache stats if enabled
        cache_stats = self.get_cache_stats()
        if cache_stats:
            stats["cache"] = cache_stats

        return stats


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Command-line interface for the daily processor."""
    import argparse

    parser = argparse.ArgumentParser(description="Polyscan Daily Processor")
    parser.add_argument("command", choices=["process", "stats", "add", "rankings", "cache", "cache-clear", "cache-prune"])
    parser.add_argument("--db", default="data/polyscan.db", help="Database path")
    parser.add_argument("--metric", type=int, help="Metric ID for rankings")
    parser.add_argument("--period", choices=TIME_PERIODS, default="all_time")
    parser.add_argument("--wallets", nargs="+", help="Wallet addresses to add")
    parser.add_argument("--force-refresh", action="store_true", help="Force API fetch (ignore cache)")
    parser.add_argument("--no-cache", action="store_true", help="Disable trade cache")
    parser.add_argument("--max-age-days", type=int, default=90, help="Max age for cache prune")

    args = parser.parse_args()

    bot = PolyscanBot(db_path=args.db, use_trade_cache=not args.no_cache)

    if args.command == "process":
        print(f"Starting daily processing of {bot.get_wallet_count():,} wallets...")

        # Show cache stats before processing
        cache_stats = bot.get_cache_stats()
        if cache_stats:
            print(f"Cache: {cache_stats['total_wallets']:,} wallets, {cache_stats['total_trades']:,} trades cached")

        def progress(done, total, stage):
            pct = (done / total) * 100 if total > 0 else 0
            print(f"\r[{stage}] {done:,}/{total:,} ({pct:.1f}%)", end="", flush=True)

        stats = bot.run_daily_processing(progress_callback=progress)
        print(f"\n\nComplete!")
        print(f"  Processed: {stats.wallets_processed:,}")
        print(f"  Failed: {stats.wallets_failed:,}")
        print(f"  Duration: {stats.duration_seconds:.1f}s")
        if stats.duration_seconds > 0:
            rate = stats.wallets_processed / stats.duration_seconds
            print(f"  Rate: {rate:,.0f} wallets/sec")

    elif args.command == "stats":
        stats = bot.get_stats()
        print(f"Total wallets: {stats['total_wallets']:,}")
        print(f"Database: {stats['db_path']}")

        if 'cache' in stats:
            cache = stats['cache']
            print(f"\nTrade Cache:")
            print(f"  Wallets cached: {cache['total_wallets']:,}")
            print(f"  Trades cached: {cache['total_trades']:,}")
            print(f"  Stale wallets: {cache['stale_wallets']:,}")
            print(f"  Cache size: {cache['cache_size_mb']:.1f} MB")
            if cache['oldest_sync']:
                print(f"  Oldest sync: {cache['oldest_sync']}")
            if cache['newest_sync']:
                print(f"  Newest sync: {cache['newest_sync']}")

    elif args.command == "add":
        if not args.wallets:
            print("Error: --wallets required")
            return
        new_count = bot.add_wallets(args.wallets)
        print(f"Added {new_count} new wallets. Total: {bot.get_wallet_count():,}")

    elif args.command == "rankings":
        if not args.metric:
            print("Error: --metric required")
            return

        rankings = bot.get_top_100(args.metric, args.period)
        print(f"Top 100 for Metric {args.metric} ({args.period}):")
        print(f"{'Rank':<6} {'Address':<44} {'Value':<15}")
        print("-" * 65)
        for rank, address, value in rankings[:20]:  # Show top 20
            print(f"{rank:<6} {address:<44} {value:<15.4f}")
        if len(rankings) > 20:
            print(f"... and {len(rankings) - 20} more")

    elif args.command == "cache":
        cache_stats = bot.get_cache_stats()
        if cache_stats:
            print("Trade Cache Statistics:")
            print(f"  Wallets cached: {cache_stats['total_wallets']:,}")
            print(f"  Trades cached: {cache_stats['total_trades']:,}")
            print(f"  Stale wallets (>24h): {cache_stats['stale_wallets']:,}")
            print(f"  Cache size: {cache_stats['cache_size_mb']:.1f} MB")
            if cache_stats['oldest_sync']:
                print(f"  Oldest sync: {cache_stats['oldest_sync']}")
            if cache_stats['newest_sync']:
                print(f"  Newest sync: {cache_stats['newest_sync']}")
        else:
            print("Trade cache is disabled")

    elif args.command == "cache-clear":
        bot.clear_cache()
        print("Trade cache cleared. Next run will fetch all data from API.")

    elif args.command == "cache-prune":
        bot.prune_cache(max_age_days=args.max_age_days)
        print(f"Pruned trades older than {args.max_age_days} days.")


if __name__ == "__main__":
    main()
