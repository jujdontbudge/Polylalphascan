"""
Parallel Deep Scanner - High-throughput wallet analysis with 1000+ wallets/sec.

Combines:
- Async HTTP fetching (200+ concurrent)
- Parallel metric calculation (multiprocessing/threading)
- Batch database writes (1000 records/batch)
- Optional ProcessPoolExecutor for true parallelism (bypasses GIL)
"""
import asyncio
import aiohttp
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing

from src.api.async_data_client import (
    AsyncDataClient, Trade, Position, WalletActivity
)
from src.database.repositories import MetricRepository, WalletRepository
from src.database.models import get_session, Wallet
from .metric_engine import MetricEngine, MultiPeriodMetricResult
from .groups.base_group import TIME_PERIODS, TIME_PERIOD_ALL

logger = logging.getLogger(__name__)

# Try to import monitoring
try:
    from src.monitoring.metrics import (
        WALLETS_SCANNED,
        SCAN_DURATION,
        THROUGHPUT,
        record_scan_result,
    )
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False


@dataclass
class ParallelScanResult:
    """Results from parallel deep scanning"""
    wallets_scanned: int = 0
    wallets_failed: int = 0
    wallets_no_trades: int = 0
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


def calculate_metrics_for_wallet(
    address: str,
    trades_data: List[Dict],
    positions_data: List[Dict],
) -> Tuple[str, Optional[Dict]]:
    """
    Calculate metrics for a single wallet (used in process pool).

    Returns tuple of (address, result_dict or None)
    """
    try:
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


class ParallelDeepScanner:
    """
    High-throughput parallel wallet scanner.

    Architecture:
    1. Async HTTP fetching with 200+ concurrent connections
    2. Parallel metric calculation with process/thread pool
    3. Batch database writes with 1000-record batches

    Target: 1000+ wallets/second throughput
    """

    def __init__(
        self,
        max_concurrent_http: int = 200,
        requests_per_second: float = 100.0,
        calc_workers: int = None,
        db_batch_size: int = 1000,
        use_process_pool: bool = False,
        enable_monitoring: bool = True,
    ):
        self.max_concurrent_http = max_concurrent_http
        self.requests_per_second = requests_per_second
        self.calc_workers = calc_workers or max(1, multiprocessing.cpu_count() - 1)
        self.db_batch_size = db_batch_size
        self.use_process_pool = use_process_pool
        self.enable_monitoring = enable_monitoring and MONITORING_AVAILABLE

        self.async_client = AsyncDataClient(
            max_concurrent=max_concurrent_http,
            requests_per_second=requests_per_second,
        )

    async def scan_wallets_async(
        self,
        wallets: List[Wallet],
        max_trades_per_wallet: int = None,
        progress_callback: callable = None,
    ) -> ParallelScanResult:
        """
        Scan multiple wallets in parallel using async/await.

        Args:
            wallets: List of Wallet objects to scan (can be detached from session)
            max_trades_per_wallet: Max trades to fetch per wallet
            progress_callback: Optional callback(stage, current, total)

        Returns:
            ParallelScanResult with metrics and stats
        """
        result = ParallelScanResult()
        result.classifications = {"bot": 0, "quant": 0, "human": 0, "unknown": 0}
        start_time = time.monotonic()

        logger.info(f"Starting parallel scan of {len(wallets)} wallets...")
        logger.info(
            f"Config: {self.max_concurrent_http} concurrent HTTP, "
            f"{self.calc_workers} calc workers, {self.db_batch_size} DB batch size"
        )

        # Build address -> wallet_id mapping
        # Access attributes safely in case objects are detached
        address_to_id = {}
        for w in wallets:
            try:
                address_to_id[w.address.lower()] = w.id
            except Exception:
                # If accessing ORM attributes fails, skip this wallet
                pass
        addresses = list(address_to_id.keys())

        # PHASE 1: Fetch all wallet data in parallel
        fetch_start = time.monotonic()
        if progress_callback:
            progress_callback("fetch", 0, len(addresses))

        activities = await self.async_client.get_multiple_wallet_activities(
            addresses,
            include_positions=True,
            max_trades_per_wallet=max_trades_per_wallet,
            progress_callback=lambda c, t: progress_callback("fetch", c, t) if progress_callback else None,
        )

        result.fetch_time = time.monotonic() - fetch_start
        logger.info(
            f"Fetch complete: {len(activities)} wallets with data "
            f"in {result.fetch_time:.1f}s ({len(activities)/max(0.1, result.fetch_time):.0f} wallets/s)"
        )

        # PHASE 2: Calculate metrics in parallel
        calc_start = time.monotonic()
        if progress_callback:
            progress_callback("calc", 0, len(activities))

        # Prepare data for process pool (serialize to dicts)
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
                        "asset_id": t.asset_id,
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
            else:
                result.wallets_no_trades += 1

        # Use ProcessPoolExecutor for true parallelism (bypasses GIL) or ThreadPoolExecutor
        calc_results = {}

        if self.use_process_pool:
            # ProcessPoolExecutor: Better for CPU-bound work, bypasses GIL
            # Use spawn context for cleaner process isolation
            ctx = multiprocessing.get_context('spawn')
            executor = ProcessPoolExecutor(max_workers=self.calc_workers, mp_context=ctx)
        else:
            # ThreadPoolExecutor: Lower overhead, but GIL-limited
            executor = ThreadPoolExecutor(max_workers=self.calc_workers)

        try:
            futures = {
                executor.submit(calculate_metrics_for_wallet, addr, trades, positions): addr
                for addr, trades, positions in calc_tasks
            }

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

        # PHASE 3: Batch save to database
        save_start = time.monotonic()
        if progress_callback:
            progress_callback("save", 0, len(calc_results))

        # Prepare batch data
        all_wallet_metrics = []
        wallet_updates = []

        for address, metrics_data in calc_results.items():
            wallet_id = address_to_id.get(address.lower())
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

            # Update stats
            result.wallets_scanned += 1
            classification = metrics_data["classification"]
            result.classifications[classification] = result.classifications.get(classification, 0) + 1

        # Batch save metrics
        session = get_session()
        try:
            with MetricRepository(session) as metric_repo:
                result.metrics_saved = metric_repo.batch_set_multiple_wallets_metrics(
                    all_wallet_metrics
                )

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

            if progress_callback:
                progress_callback("save", len(calc_results), len(calc_results))

        finally:
            session.close()

        result.save_time = time.monotonic() - save_start
        result.wallets_failed = len(wallets) - result.wallets_scanned - result.wallets_no_trades

        result.duration_seconds = time.monotonic() - start_time

        # Update Prometheus metrics if monitoring enabled
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
            f"Parallel scan complete: {result.wallets_scanned} scanned, "
            f"{result.wallets_failed} failed, {result.wallets_no_trades} no trades\n"
            f"  Metrics: {result.metrics_calculated} calculated, {result.metrics_saved} saved\n"
            f"  Performance: {result.wallets_per_second:.1f} wallets/s\n"
            f"  Timing: fetch={result.fetch_time:.1f}s, calc={result.calc_time:.1f}s, save={result.save_time:.1f}s"
        )

        return result

    def scan_wallets(
        self,
        wallets: List[Wallet],
        max_trades_per_wallet: int = None,
        progress_callback: callable = None,
    ) -> ParallelScanResult:
        """
        Synchronous wrapper for scan_wallets_async.

        Args:
            wallets: List of Wallet objects
            max_trades_per_wallet: Max trades per wallet
            progress_callback: Optional callback(stage, current, total)

        Returns:
            ParallelScanResult
        """
        return asyncio.run(
            self.scan_wallets_async(wallets, max_trades_per_wallet, progress_callback)
        )

    def scan_unscanned_wallets(
        self,
        limit: int = 1000,
        max_trades_per_wallet: int = None,
        progress_callback: callable = None,
    ) -> ParallelScanResult:
        """
        Scan wallets that haven't been deep scanned yet.

        Args:
            limit: Maximum wallets to scan
            max_trades_per_wallet: Max trades per wallet
            progress_callback: Optional callback

        Returns:
            ParallelScanResult
        """
        session = get_session()
        try:
            with WalletRepository(session) as repo:
                wallets = repo.get_unscanned(limit=limit)
        finally:
            session.close()

        if not wallets:
            logger.info("No unscanned wallets found")
            return ParallelScanResult()

        return self.scan_wallets(wallets, max_trades_per_wallet, progress_callback)


class UltraScanner:
    """
    Ultra high-throughput scanner combining harvesting + analysis.

    Designed for scanning 10,000+ wallets per minute.
    """

    def __init__(
        self,
        max_concurrent: int = 300,
        requests_per_second: float = 150.0,
    ):
        from src.core.scanner.parallel_harvester import UltraHarvester

        self.harvester = UltraHarvester(
            max_concurrent=max_concurrent,
            requests_per_second=requests_per_second,
        )
        self.scanner = ParallelDeepScanner(
            max_concurrent_http=max_concurrent,
            requests_per_second=requests_per_second,
        )

    async def discover_and_scan(
        self,
        min_volume: float = 500.0,
        max_markets: int = 1000,
        max_wallets: int = 10000,
        progress_callback: callable = None,
    ) -> Tuple[Any, ParallelScanResult]:
        """
        Discover wallets from markets and scan them.

        Args:
            min_volume: Minimum market volume
            max_markets: Maximum markets to scan
            max_wallets: Maximum wallets to analyze
            progress_callback: Optional callback

        Returns:
            Tuple of (harvest_result, scan_result)
        """
        # Phase 1: Harvest wallets from markets
        logger.info(f"Phase 1: Harvesting wallets from {max_markets} markets...")
        harvest_result = await self.harvester.harvest_all(
            min_volume=min_volume,
            max_markets=max_markets,
        )

        if not harvest_result.wallets:
            logger.warning("No wallets harvested")
            return harvest_result, ParallelScanResult()

        # Get or create wallets in database
        wallet_addresses = list(harvest_result.wallets)[:max_wallets]
        logger.info(f"Phase 2: Processing {len(wallet_addresses)} wallets...")

        session = get_session()
        try:
            with WalletRepository(session) as repo:
                wallets = []
                for address in wallet_addresses:
                    wallet = repo.get_or_create(address)
                    wallets.append(wallet)
        finally:
            session.close()

        # Phase 3: Deep scan wallets
        logger.info(f"Phase 3: Deep scanning {len(wallets)} wallets...")
        scan_result = await self.scanner.scan_wallets_async(
            wallets,
            progress_callback=progress_callback,
        )

        return harvest_result, scan_result

    def discover_and_scan_sync(
        self,
        min_volume: float = 500.0,
        max_markets: int = 1000,
        max_wallets: int = 10000,
        progress_callback: callable = None,
    ) -> Tuple[Any, ParallelScanResult]:
        """Synchronous wrapper"""
        return asyncio.run(
            self.discover_and_scan(min_volume, max_markets, max_wallets, progress_callback)
        )


# Convenience function
def scan_1000_wallets(
    max_concurrent: int = 200,
    requests_per_second: float = 100.0,
) -> ParallelScanResult:
    """
    Convenience function to scan 1000 unscanned wallets.

    Returns:
        ParallelScanResult with metrics
    """
    scanner = ParallelDeepScanner(
        max_concurrent_http=max_concurrent,
        requests_per_second=requests_per_second,
    )
    return scanner.scan_unscanned_wallets(limit=1000)
