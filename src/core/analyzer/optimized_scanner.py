"""
Optimized Scanner - High-throughput wallet analysis with all Phase 1-4 optimizations.

Combines:
- Phase 1: Cost basis caching, vectorized calculations, parallel fetch
- Phase 2: Single-pass multi-period, Numba JIT, ProcessPoolExecutor
- Phase 3: Unified data router (Goldsky, HyperSync, REST), local trade index
- Phase 4: Redis caching, Prometheus monitoring, Celery-ready

Target: 1000+ wallets/second sustained throughput.
"""
import asyncio
import logging
import time
import os
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing

from src.api.async_data_client import Trade, Position, WalletActivity

logger = logging.getLogger(__name__)

# Try to import optional components
try:
    from src.api.unified_data_router import UnifiedDataRouter
    UNIFIED_ROUTER_AVAILABLE = True
except ImportError:
    UNIFIED_ROUTER_AVAILABLE = False
    logger.debug("UnifiedDataRouter not available")

try:
    from src.cache.redis_cache import MetricCache, get_cache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    logger.debug("Redis cache not available")

try:
    from src.core.indexer.trade_indexer import TradeIndexer, get_indexer
    INDEXER_AVAILABLE = True
except ImportError:
    INDEXER_AVAILABLE = False
    logger.debug("Trade indexer not available")

try:
    from src.monitoring.metrics import (
        WALLETS_SCANNED,
        SCAN_DURATION,
        THROUGHPUT,
        METRICS_CALCULATED,
        CACHE_HITS,
        CACHE_MISSES,
        record_scan_result,
        MetricsContext,
    )
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    logger.debug("Prometheus monitoring not available")


@dataclass
class OptimizedScanResult:
    """Results from optimized scanning"""
    wallets_scanned: int = 0
    wallets_failed: int = 0
    wallets_no_trades: int = 0
    wallets_from_cache: int = 0
    total_trades_processed: int = 0
    metrics_calculated: int = 0
    metrics_saved: int = 0
    classifications: Dict[str, int] = field(default_factory=dict)
    duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)

    # Performance stats
    fetch_time: float = 0.0
    calc_time: float = 0.0
    save_time: float = 0.0

    # Data source stats
    data_source_stats: Dict[str, int] = field(default_factory=dict)

    @property
    def wallets_per_second(self) -> float:
        if self.duration_seconds > 0:
            return self.wallets_scanned / self.duration_seconds
        return 0.0

    @property
    def success_rate(self) -> float:
        total = self.wallets_scanned + self.wallets_failed
        if total > 0:
            return self.wallets_scanned / total * 100
        return 0.0

    @property
    def cache_hit_rate(self) -> float:
        total = self.wallets_scanned + self.wallets_from_cache
        if total > 0:
            return self.wallets_from_cache / total * 100
        return 0.0


def _calculate_metrics_worker(args: Tuple[str, List[Dict], List[Dict]]) -> Tuple[str, Optional[Dict]]:
    """
    Worker function for ProcessPoolExecutor.

    Calculates metrics for a single wallet. Designed to run in separate process
    to bypass GIL for CPU-bound work.
    """
    address, trades_data, positions_data = args

    try:
        from src.api.async_data_client import Trade, Position
        from src.core.analyzer.metric_engine import MetricEngine

        # Reconstruct Trade/Position objects from dicts
        trades = [Trade.from_dict(t) for t in trades_data]
        positions = [Position.from_dict(p) for p in positions_data]

        if not trades:
            return address, None

        # Create engine and calculate
        engine = MetricEngine()
        result = engine.calculate_all_periods(address, trades, positions)

        # Convert to serializable dict
        return address, {
            "results": {
                period: {
                    "metrics": r.metrics,
                    "quant_score": r.quant_score,
                    "classification": r.classification,
                }
                for period, r in result.results.items()
            },
            "classification": result.classification,
            "quant_score": result.quant_score,
            "total_calculated": result.total_calculated,
            "trade_count": len(trades),
            "volume": sum(t.value for t in trades),
        }
    except Exception as e:
        logger.debug(f"Error calculating metrics for {address}: {e}")
        return address, None


class OptimizedScanner:
    """
    Ultra high-throughput wallet scanner with all optimizations.

    Integrates all Phase 1-4 optimizations:
    - Smart data routing (Goldsky > REST with caching)
    - ProcessPoolExecutor for CPU-bound calculations
    - Redis caching for repeated lookups
    - Prometheus metrics for monitoring
    - Local trade index for instant lookups

    Target: 1000+ wallets/second sustained throughput
    """

    def __init__(
        self,
        max_concurrent_http: int = 200,
        requests_per_second: float = 100.0,
        calc_workers: int = None,
        db_batch_size: int = 1000,
        use_process_pool: bool = True,
        enable_cache: bool = True,
        enable_indexer: bool = True,
        enable_monitoring: bool = True,
        cache_ttl_seconds: int = 3600,
        redis_url: str = "redis://localhost:6379",
    ):
        self.max_concurrent_http = max_concurrent_http
        self.requests_per_second = requests_per_second
        self.calc_workers = calc_workers or max(1, multiprocessing.cpu_count() - 1)
        self.db_batch_size = db_batch_size
        self.use_process_pool = use_process_pool

        # Initialize data router
        if UNIFIED_ROUTER_AVAILABLE:
            self.data_router = UnifiedDataRouter(
                cache_ttl_seconds=cache_ttl_seconds,
                prefer_batch=True,
            )
        else:
            self.data_router = None
            from src.api.async_data_client import AsyncDataClient
            self.async_client = AsyncDataClient(
                max_concurrent=max_concurrent_http,
                requests_per_second=requests_per_second,
            )

        # Initialize cache
        if enable_cache and CACHE_AVAILABLE:
            self.cache = get_cache(redis_url)
        else:
            self.cache = None

        # Initialize trade indexer
        if enable_indexer and INDEXER_AVAILABLE:
            self.indexer = get_indexer()
        else:
            self.indexer = None

        # Monitoring flag
        self.enable_monitoring = enable_monitoring and MONITORING_AVAILABLE

    async def scan_addresses_async(
        self,
        addresses: List[str],
        max_trades_per_wallet: int = None,
        progress_callback: callable = None,
        skip_cache_check: bool = False,
    ) -> OptimizedScanResult:
        """
        Scan wallet addresses with all optimizations.

        Args:
            addresses: List of wallet addresses to scan
            max_trades_per_wallet: Max trades to fetch per wallet
            progress_callback: Optional callback(stage, current, total)
            skip_cache_check: Skip cache lookup (for benchmarking)

        Returns:
            OptimizedScanResult with metrics and stats
        """
        result = OptimizedScanResult()
        result.classifications = {"bot": 0, "quant": 0, "human": 0, "unknown": 0}
        result.data_source_stats = {"cache": 0, "goldsky": 0, "rest": 0, "indexer": 0}
        start_time = time.monotonic()

        logger.info(f"Starting optimized scan of {len(addresses)} wallets...")
        logger.info(
            f"Config: {self.max_concurrent_http} concurrent HTTP, "
            f"{self.calc_workers} calc workers, "
            f"cache={'ON' if self.cache else 'OFF'}, "
            f"indexer={'ON' if self.indexer else 'OFF'}"
        )

        # Normalize addresses
        addresses = [addr.lower() for addr in addresses]

        # PHASE 0: Check cache for already-computed metrics
        cached_results = {}
        uncached_addresses = addresses

        if self.cache and not skip_cache_check:
            logger.info("Checking cache for pre-computed metrics...")
            cached_metrics = self.cache.batch_get_metrics(addresses)

            for addr, metrics in cached_metrics.items():
                if metrics:
                    cached_results[addr] = metrics
                    result.wallets_from_cache += 1
                    result.data_source_stats["cache"] += 1

            uncached_addresses = [a for a in addresses if a not in cached_results]
            logger.info(f"  Cache: {len(cached_results)} hits, {len(uncached_addresses)} to fetch")

            if self.enable_monitoring:
                CACHE_HITS.labels(cache_type="metrics").inc(len(cached_results))
                CACHE_MISSES.labels(cache_type="metrics").inc(len(uncached_addresses))

        if not uncached_addresses:
            result.wallets_scanned = len(cached_results)
            result.duration_seconds = time.monotonic() - start_time
            return result

        # PHASE 1: Fetch wallet data using unified router
        fetch_start = time.monotonic()
        if progress_callback:
            progress_callback("fetch", 0, len(uncached_addresses))

        if self.data_router:
            activities = await self.data_router.get_multiple_wallets(
                uncached_addresses,
                include_positions=True,
                progress_callback=lambda c, t: progress_callback("fetch", c, t) if progress_callback else None,
            )
            # Update data source stats
            stats = self.data_router.get_stats_summary()
            for source in ["goldsky", "rest"]:
                if source in stats:
                    result.data_source_stats[source] = stats[source].get("requests", 0)
        else:
            activities = await self.async_client.get_multiple_wallet_activities(
                uncached_addresses,
                include_positions=True,
                max_trades_per_wallet=max_trades_per_wallet,
                progress_callback=lambda c, t: progress_callback("fetch", c, t) if progress_callback else None,
            )
            result.data_source_stats["rest"] = len(activities)

        result.fetch_time = time.monotonic() - fetch_start
        logger.info(
            f"Fetch complete: {len(activities)} wallets with data "
            f"in {result.fetch_time:.1f}s ({len(activities)/max(0.1, result.fetch_time):.0f} wallets/s)"
        )

        # PHASE 2: Calculate metrics with ProcessPoolExecutor
        calc_start = time.monotonic()
        if progress_callback:
            progress_callback("calc", 0, len(activities))

        # Prepare data for worker pool
        calc_tasks = []
        for address, activity in activities.items():
            if activity.trades:
                trades_data = [
                    {
                        "id": t.id,
                        "transaction_hash": t.transaction_hash,
                        "timestamp": t.timestamp.isoformat() if t.timestamp else None,
                        "market_id": t.market_id,
                        "condition_id": t.condition_id,
                        "wallet_address": t.wallet_address,
                        "side": t.side,
                        "outcome": t.outcome,
                        "price": t.price,
                        "size": t.size,
                        "value": t.value,
                        "fee": t.fee,
                        "asset_id": getattr(t, 'asset_id', None),
                    }
                    for t in activity.trades
                ]
                positions_data = [
                    {
                        "market_id": p.market_id,
                        "condition_id": p.condition_id,
                        "wallet_address": p.wallet_address,
                        "outcome": p.outcome,
                        "outcomeIndex": p.outcome_index,
                        "size": p.size,
                        "avgPrice": p.avg_price,
                        "currentPrice": p.current_price,
                        "realizedPnl": p.realized_pnl,
                    }
                    for p in activity.positions
                ]
                calc_tasks.append((address, trades_data, positions_data))
                result.total_trades_processed += len(activity.trades)

                # Update trade indexer
                if self.indexer:
                    await self._update_indexer(address, trades_data)
            else:
                result.wallets_no_trades += 1

        # Use ProcessPoolExecutor for true parallelism (bypasses GIL)
        calc_results = {}
        executor_class = ProcessPoolExecutor if self.use_process_pool else ThreadPoolExecutor

        # Use spawn context for cleaner process isolation
        if self.use_process_pool:
            ctx = multiprocessing.get_context('spawn')
            executor = ProcessPoolExecutor(max_workers=self.calc_workers, mp_context=ctx)
        else:
            executor = ThreadPoolExecutor(max_workers=self.calc_workers)

        try:
            # Submit all tasks
            futures = {executor.submit(_calculate_metrics_worker, task): task[0] for task in calc_tasks}

            completed = 0
            for future in as_completed(futures):
                try:
                    address, metrics_data = future.result(timeout=120)
                    if metrics_data:
                        calc_results[address] = metrics_data
                        result.metrics_calculated += metrics_data["total_calculated"]
                except Exception as e:
                    logger.debug(f"Calc error: {e}")

                completed += 1
                if progress_callback and completed % 50 == 0:
                    progress_callback("calc", completed, len(calc_tasks))
        finally:
            executor.shutdown(wait=False)

        result.calc_time = time.monotonic() - calc_start
        logger.info(
            f"Calc complete: {len(calc_results)} wallets processed "
            f"in {result.calc_time:.1f}s ({len(calc_results)/max(0.1, result.calc_time):.0f} wallets/s)"
        )

        # PHASE 3: Update cache with new results
        if self.cache and calc_results:
            cache_data = {}
            for address, metrics_data in calc_results.items():
                for time_period, period_data in metrics_data["results"].items():
                    cache_key = f"{address}:{time_period}"
                    cache_data[address] = period_data["metrics"]
            self.cache.batch_set_metrics(cache_data)

        # PHASE 4: Save to database
        save_start = time.monotonic()
        if progress_callback:
            progress_callback("save", 0, len(calc_results))

        result.metrics_saved = await self._save_results(calc_results, result)

        result.save_time = time.monotonic() - save_start
        result.wallets_scanned = len(calc_results) + result.wallets_from_cache
        result.wallets_failed = len(addresses) - result.wallets_scanned - result.wallets_no_trades
        result.duration_seconds = time.monotonic() - start_time

        # Update Prometheus metrics
        if self.enable_monitoring:
            record_scan_result(
                wallets_scanned=result.wallets_scanned,
                wallets_failed=result.wallets_failed,
                wallets_no_trades=result.wallets_no_trades,
                metrics_calculated=result.metrics_calculated,
                metrics_saved=result.metrics_saved,
                classifications=result.classifications,
                duration_seconds=result.duration_seconds,
            )

        logger.info(
            f"Optimized scan complete: {result.wallets_scanned} scanned "
            f"({result.wallets_from_cache} from cache), "
            f"{result.wallets_failed} failed, {result.wallets_no_trades} no trades\n"
            f"  Metrics: {result.metrics_calculated} calculated, {result.metrics_saved} saved\n"
            f"  Performance: {result.wallets_per_second:.1f} wallets/s\n"
            f"  Timing: fetch={result.fetch_time:.1f}s, calc={result.calc_time:.1f}s, save={result.save_time:.1f}s\n"
            f"  Data sources: {result.data_source_stats}"
        )

        return result

    async def _update_indexer(self, address: str, trades_data: List[Dict]):
        """Update local trade index with new trades"""
        if not self.indexer:
            return

        try:
            for trade in trades_data:
                trade["wallet_address"] = address
            await self.indexer.bulk_insert_trades_async(trades_data, source="scan")
        except Exception as e:
            logger.debug(f"Failed to update indexer: {e}")

    async def _save_results(
        self,
        calc_results: Dict[str, Dict],
        result: OptimizedScanResult,
    ) -> int:
        """Save calculated results to database"""
        from src.database.repositories import MetricRepository, WalletRepository
        from src.database.models import get_session

        # First, get or create wallet IDs
        session = get_session()
        try:
            with WalletRepository(session) as wallet_repo:
                address_to_id = {}
                for address in calc_results.keys():
                    wallet = wallet_repo.get_or_create(address)
                    address_to_id[address] = wallet.id

            # Prepare batch data
            all_wallet_metrics = []
            wallet_updates = []

            for address, metrics_data in calc_results.items():
                wallet_id = address_to_id.get(address)
                if not wallet_id:
                    continue

                # Prepare metrics for each time period
                for time_period, period_data in metrics_data["results"].items():
                    all_wallet_metrics.append({
                        "wallet_id": wallet_id,
                        "metrics": period_data["metrics"],
                        "time_period": time_period,
                    })

                # Prepare wallet classification update
                wallet_updates.append({
                    "wallet_id": wallet_id,
                    "classification": metrics_data["classification"],
                    "quant_score": metrics_data["quant_score"],
                    "trade_count": metrics_data["trade_count"],
                    "volume": metrics_data["volume"],
                })

                # Update result stats
                classification = metrics_data["classification"]
                result.classifications[classification] = result.classifications.get(classification, 0) + 1

            # Batch save metrics
            with MetricRepository(session) as metric_repo:
                metrics_saved = metric_repo.batch_set_multiple_wallets_metrics(all_wallet_metrics)

            with WalletRepository(session) as wallet_repo:
                for update in wallet_updates:
                    wallet_repo.set_classification(
                        update["wallet_id"],
                        update["classification"],
                        update["quant_score"],
                    )
                    wallet_repo.mark_scanned(
                        update["wallet_id"],
                        total_trades=update["trade_count"],
                        total_volume=update["volume"],
                    )

            return metrics_saved
        finally:
            session.close()

    def scan_addresses(
        self,
        addresses: List[str],
        max_trades_per_wallet: int = None,
        progress_callback: callable = None,
    ) -> OptimizedScanResult:
        """Synchronous wrapper for scan_addresses_async"""
        return asyncio.run(
            self.scan_addresses_async(addresses, max_trades_per_wallet, progress_callback)
        )

    async def scan_wallets_async(
        self,
        wallets: List[Any],
        max_trades_per_wallet: int = None,
        progress_callback: callable = None,
    ) -> OptimizedScanResult:
        """
        Scan Wallet objects (compatible with ParallelDeepScanner interface).
        """
        addresses = []
        for w in wallets:
            try:
                addresses.append(w.address.lower())
            except Exception:
                pass

        return await self.scan_addresses_async(
            addresses, max_trades_per_wallet, progress_callback
        )

    def scan_wallets(
        self,
        wallets: List[Any],
        max_trades_per_wallet: int = None,
        progress_callback: callable = None,
    ) -> OptimizedScanResult:
        """Synchronous wrapper for scan_wallets_async"""
        return asyncio.run(
            self.scan_wallets_async(wallets, max_trades_per_wallet, progress_callback)
        )

    def scan_unscanned_wallets(
        self,
        limit: int = 1000,
        max_trades_per_wallet: int = None,
        progress_callback: callable = None,
    ) -> OptimizedScanResult:
        """Scan wallets that haven't been deep scanned yet"""
        from src.database.repositories import WalletRepository
        from src.database.models import get_session

        session = get_session()
        try:
            with WalletRepository(session) as repo:
                wallets = repo.get_unscanned(limit=limit)
        finally:
            session.close()

        if not wallets:
            logger.info("No unscanned wallets found")
            return OptimizedScanResult()

        return self.scan_wallets(wallets, max_trades_per_wallet, progress_callback)

    def get_status(self) -> Dict[str, Any]:
        """Get scanner status and statistics"""
        status = {
            "config": {
                "max_concurrent_http": self.max_concurrent_http,
                "requests_per_second": self.requests_per_second,
                "calc_workers": self.calc_workers,
                "use_process_pool": self.use_process_pool,
            },
            "components": {
                "data_router": UNIFIED_ROUTER_AVAILABLE and self.data_router is not None,
                "cache": CACHE_AVAILABLE and self.cache is not None,
                "indexer": INDEXER_AVAILABLE and self.indexer is not None,
                "monitoring": self.enable_monitoring,
            },
        }

        if self.cache:
            status["cache_stats"] = self.cache.stats()

        if self.indexer:
            status["indexer_stats"] = str(self.indexer.get_stats())

        if self.data_router:
            status["data_router_stats"] = self.data_router.get_stats_summary()

        return status


class UltraOptimizedScanner(OptimizedScanner):
    """
    Maximum throughput scanner with aggressive optimizations.

    Uses:
    - 300 concurrent HTTP connections
    - 150 requests/second
    - All CPU cores for calculation
    - Aggressive caching
    """

    def __init__(self, **kwargs):
        defaults = {
            "max_concurrent_http": 300,
            "requests_per_second": 150.0,
            "calc_workers": multiprocessing.cpu_count(),
            "use_process_pool": True,
            "enable_cache": True,
            "enable_indexer": True,
            "enable_monitoring": True,
        }
        defaults.update(kwargs)
        super().__init__(**defaults)


# Convenience functions
def scan_1000_wallets_optimized() -> OptimizedScanResult:
    """Scan 1000 unscanned wallets with all optimizations"""
    scanner = OptimizedScanner()
    return scanner.scan_unscanned_wallets(limit=1000)


async def scan_addresses_optimized(
    addresses: List[str],
    max_concurrent: int = 200,
) -> OptimizedScanResult:
    """Scan addresses with optimized scanner"""
    scanner = OptimizedScanner(max_concurrent_http=max_concurrent)
    return await scanner.scan_addresses_async(addresses)
