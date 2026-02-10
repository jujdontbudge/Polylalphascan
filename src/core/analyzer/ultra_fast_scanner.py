"""
Ultra Fast Scanner - 50,000+ wallets/second using pure Numba metrics.

Bypasses pandas entirely for maximum throughput.

Supports two modes:
- Fast mode (30 metrics): 110,000+ wallets/second
- Full mode (117 metrics): 50,000+ wallets/second
"""
import asyncio
import time
import logging
from typing import List, Dict, Optional, Tuple, Literal
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

from src.api.async_data_client import AsyncDataClient

# Import both fast (30 metrics) and full (117 metrics) calculators
from src.core.analyzer.fast_metrics import (
    FastWalletData,
    calculate_fast_metrics,
    FAST_METRIC_IDS,
)
from src.core.analyzer.numba_metrics import (
    FastWalletData as FastWalletData117,
    calculate_117_metrics,
    calculate_quant_score,
    calculate_batch_117_parallel,
    METRIC_NAMES,
)

logger = logging.getLogger(__name__)

# Metric mode type
MetricMode = Literal["fast", "full"]


@dataclass
class UltraFastResult:
    """Results from ultra-fast scan."""
    wallets_scanned: int = 0
    wallets_failed: int = 0
    wallets_no_trades: int = 0
    metrics_calculated: int = 0
    total_trades_processed: int = 0

    fetch_time: float = 0.0
    calc_time: float = 0.0
    total_time: float = 0.0

    # Metric mode used
    metric_mode: str = "fast"  # "fast" (30 metrics) or "full" (117 metrics)

    # Results
    wallet_metrics: Dict[str, Dict[int, float]] = field(default_factory=dict)

    # Quant scores (only populated in full mode)
    wallet_quant_scores: Dict[str, float] = field(default_factory=dict)

    @property
    def wallets_per_second(self) -> float:
        if self.total_time <= 0:
            return 0.0
        return self.wallets_scanned / self.total_time

    @property
    def calc_wallets_per_second(self) -> float:
        if self.calc_time <= 0:
            return 0.0
        return self.wallets_scanned / self.calc_time

    @property
    def metrics_per_wallet(self) -> int:
        """Number of metrics calculated per wallet."""
        return 117 if self.metric_mode == "full" else 30


def calculate_wallet_metrics_fast(args: Tuple[str, List[dict]]) -> Tuple[str, Dict[int, float], int]:
    """
    Calculate 30 fast metrics for a single wallet (for process pool).

    Returns (address, metrics_dict, trade_count)
    """
    address, trades = args
    if not trades:
        return address, {}, 0

    data = FastWalletData.from_trades(address, trades)
    metrics = calculate_fast_metrics(data)
    return address, metrics, len(trades)


def calculate_wallet_metrics_full(args: Tuple[str, List[dict]]) -> Tuple[str, Dict[int, float], int, float]:
    """
    Calculate all 117 metrics for a single wallet (for process pool).

    Returns (address, metrics_dict, trade_count, quant_score)
    """
    address, trades = args
    if not trades:
        return address, {}, 0, 0.0

    data = FastWalletData117.from_trades(address, trades)
    metrics = calculate_117_metrics(data)
    quant_score = calculate_quant_score(metrics)
    return address, metrics, len(trades), quant_score


# Keep backwards compatibility alias
calculate_wallet_metrics = calculate_wallet_metrics_fast


class UltraFastScanner:
    """
    Ultra-fast wallet scanner using pure Numba metrics.

    Target speeds:
    - Fast mode (30 metrics): 110,000+ wallets/second
    - Full mode (117 metrics): 50,000+ wallets/second

    Architecture:
    1. Async HTTP fetching (same as before)
    2. Pure Numba metric calculation (no pandas)
    3. Optional ProcessPool for multi-core scaling
    """

    def __init__(
        self,
        max_concurrent_http: int = 200,
        requests_per_second: float = 100.0,
        use_multiprocessing: bool = False,
        workers: int = None,
        metric_mode: MetricMode = "fast",
    ):
        """
        Initialize the ultra-fast scanner.

        Args:
            max_concurrent_http: Max concurrent HTTP connections
            requests_per_second: Rate limit for API calls
            use_multiprocessing: Use ProcessPoolExecutor for calculation
            workers: Number of worker processes (default: CPU count - 1)
            metric_mode: "fast" for 30 metrics, "full" for all 117 metrics
        """
        self.max_concurrent_http = max_concurrent_http
        self.requests_per_second = requests_per_second
        self.use_multiprocessing = use_multiprocessing
        self.workers = workers or max(1, multiprocessing.cpu_count() - 1)
        self.metric_mode = metric_mode

        self.async_client = AsyncDataClient(
            max_concurrent=max_concurrent_http,
            requests_per_second=requests_per_second,
        )

    async def scan_addresses_async(
        self,
        addresses: List[str],
        max_trades_per_wallet: int = 1000,
        progress_callback: callable = None,
        metric_mode: MetricMode = None,
    ) -> UltraFastResult:
        """
        Scan wallet addresses with ultra-fast metric calculation.

        Args:
            addresses: List of wallet addresses to scan
            max_trades_per_wallet: Max trades to fetch per wallet
            progress_callback: Optional callback(stage, current, total)
            metric_mode: Override the scanner's default metric mode

        Returns:
            UltraFastResult with metrics for all wallets
        """
        mode = metric_mode or self.metric_mode
        result = UltraFastResult(metric_mode=mode)
        start_time = time.monotonic()

        # PHASE 1: Fetch trade data
        fetch_start = time.monotonic()
        if progress_callback:
            progress_callback("fetch", 0, len(addresses))

        # Fetch all wallet activities
        activities = await self.async_client.get_multiple_wallet_activities(
            addresses,
            include_positions=False,  # Skip positions for speed
            max_trades_per_wallet=max_trades_per_wallet,
            progress_callback=lambda c, t: progress_callback("fetch", c, t) if progress_callback else None,
        )

        result.fetch_time = time.monotonic() - fetch_start
        logger.info(f"Fetch complete: {len(activities)} wallets in {result.fetch_time:.1f}s")

        # PHASE 2: Calculate metrics (ultra-fast)
        calc_start = time.monotonic()
        if progress_callback:
            progress_callback("calc", 0, len(activities))

        # Prepare data for calculation
        calc_tasks = []
        for address, activity in activities.items():
            if activity.trades:
                trades_data = [
                    {
                        'price': t.price,
                        'size': t.size,
                        'fee': t.fee,
                        'side': t.side,
                        'timestamp': t.timestamp,
                        'market_id': t.market_id or t.condition_id,
                        'value': t.value,
                    }
                    for t in activity.trades
                ]
                calc_tasks.append((address, trades_data))
                result.total_trades_processed += len(activity.trades)
            else:
                result.wallets_no_trades += 1

        # Select calculator based on mode
        use_full = mode == "full"
        calc_func = calculate_wallet_metrics_full if use_full else calculate_wallet_metrics_fast

        # Calculate metrics
        if use_full and len(calc_tasks) >= 100:
            # Use Numba parallel batch for large 117-metric batches
            # This is faster than ProcessPool due to less overhead
            try:
                addresses = [addr for addr, _ in calc_tasks]
                wallet_data = [FastWalletData117.from_trades(addr, trades) for addr, trades in calc_tasks]

                # Parallel batch calculation
                metrics_list = calculate_batch_117_parallel(wallet_data)

                for i, (address, metrics) in enumerate(zip(addresses, metrics_list)):
                    result.wallet_metrics[address] = metrics
                    result.wallet_quant_scores[address] = calculate_quant_score(metrics)
                    result.metrics_calculated += len(metrics)
                    result.wallets_scanned += 1

                    if progress_callback and (i + 1) % 100 == 0:
                        progress_callback("calc", i + 1, len(calc_tasks))

            except Exception as e:
                logger.warning(f"Parallel batch failed, falling back to sequential: {e}")
                # Fallback to sequential
                for i, (address, trades) in enumerate(calc_tasks):
                    try:
                        data = FastWalletData117.from_trades(address, trades)
                        metrics = calculate_117_metrics(data)
                        result.wallet_quant_scores[address] = calculate_quant_score(metrics)
                        result.wallet_metrics[address] = metrics
                        result.metrics_calculated += len(metrics)
                        result.wallets_scanned += 1
                    except Exception as e2:
                        logger.debug(f"Error calculating metrics for {address}: {e2}")
                        result.wallets_failed += 1

        elif self.use_multiprocessing and len(calc_tasks) > 10:
            # Use ProcessPool for large batches (fast mode)
            ctx = multiprocessing.get_context('spawn')
            with ProcessPoolExecutor(max_workers=self.workers, mp_context=ctx) as executor:
                futures = list(executor.map(calc_func, calc_tasks))

            for item in futures:
                if use_full:
                    address, metrics, trade_count, quant_score = item
                    if metrics:
                        result.wallet_quant_scores[address] = quant_score
                else:
                    address, metrics, trade_count = item

                if metrics:
                    result.wallet_metrics[address] = metrics
                    result.metrics_calculated += len(metrics)
                    result.wallets_scanned += 1
                else:
                    result.wallets_failed += 1
        else:
            # Single-threaded (still fast due to Numba)
            for i, (address, trades) in enumerate(calc_tasks):
                try:
                    if use_full:
                        data = FastWalletData117.from_trades(address, trades)
                        metrics = calculate_117_metrics(data)
                        result.wallet_quant_scores[address] = calculate_quant_score(metrics)
                    else:
                        data = FastWalletData.from_trades(address, trades)
                        metrics = calculate_fast_metrics(data)

                    result.wallet_metrics[address] = metrics
                    result.metrics_calculated += len(metrics)
                    result.wallets_scanned += 1
                except Exception as e:
                    logger.debug(f"Error calculating metrics for {address}: {e}")
                    result.wallets_failed += 1

                if progress_callback and (i + 1) % 100 == 0:
                    progress_callback("calc", i + 1, len(calc_tasks))

        result.calc_time = time.monotonic() - calc_start
        result.total_time = time.monotonic() - start_time

        metrics_label = "117" if use_full else "30"
        logger.info(
            f"Scan complete ({metrics_label} metrics): {result.wallets_scanned} wallets in {result.total_time:.1f}s "
            f"({result.wallets_per_second:.0f} wallets/sec total, "
            f"{result.calc_wallets_per_second:.0f} wallets/sec calc)"
        )

        return result

    def scan_addresses(
        self,
        addresses: List[str],
        max_trades_per_wallet: int = 1000,
        progress_callback: callable = None,
        metric_mode: MetricMode = None,
    ) -> UltraFastResult:
        """Synchronous wrapper for scan_addresses_async."""
        return asyncio.run(
            self.scan_addresses_async(addresses, max_trades_per_wallet, progress_callback, metric_mode)
        )


# Convenience functions
def ultra_fast_scan(
    addresses: List[str],
    max_concurrent: int = 200,
    requests_per_second: float = 100.0,
    metric_mode: MetricMode = "fast",
) -> UltraFastResult:
    """
    Convenience function for ultra-fast wallet scanning.

    Args:
        addresses: Wallet addresses to scan
        max_concurrent: Max concurrent HTTP connections
        requests_per_second: Rate limit
        metric_mode: "fast" for 30 metrics, "full" for 117 metrics

    Returns:
        UltraFastResult with all metrics
    """
    scanner = UltraFastScanner(
        max_concurrent_http=max_concurrent,
        requests_per_second=requests_per_second,
        metric_mode=metric_mode,
    )
    return scanner.scan_addresses(addresses)


def ultra_fast_scan_117(
    addresses: List[str],
    max_concurrent: int = 200,
    requests_per_second: float = 100.0,
) -> UltraFastResult:
    """
    Convenience function for ultra-fast scanning with all 117 metrics.

    Target: 50,000+ wallets/second calculation speed.

    Args:
        addresses: Wallet addresses to scan
        max_concurrent: Max concurrent HTTP connections
        requests_per_second: Rate limit

    Returns:
        UltraFastResult with all 117 metrics and quant scores
    """
    return ultra_fast_scan(addresses, max_concurrent, requests_per_second, metric_mode="full")


# =============================================================================
# MEMORY-SAFE STREAMING SCANNER
# =============================================================================

@dataclass
class StreamingResult:
    """Results from streaming scan with memory stats."""
    total_wallets: int = 0
    wallets_processed: int = 0
    wallets_failed: int = 0
    wallets_no_trades: int = 0
    chunks_processed: int = 0

    total_time: float = 0.0
    peak_memory_mb: float = 0.0

    # Results stored in Redis/callback, not in memory
    results_stored_to: str = "memory"  # "memory", "redis", "callback"


class StreamingScanner:
    """
    Memory-safe streaming scanner for processing unlimited wallets.

    Processes wallets in chunks to maintain constant memory usage.
    Results are streamed to Redis or a callback function instead of
    accumulating in RAM.

    Memory usage: ~40-100 MB regardless of total wallet count.
    """

    DEFAULT_CHUNK_SIZE = 1000  # Wallets per chunk

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        max_concurrent_http: int = 200,
        requests_per_second: float = 100.0,
        redis_url: Optional[str] = None,
        metric_mode: MetricMode = "full",
    ):
        """
        Initialize streaming scanner.

        Args:
            chunk_size: Number of wallets to process per chunk (default: 1000)
            max_concurrent_http: Max concurrent HTTP connections
            requests_per_second: Rate limit for API calls
            redis_url: Redis URL for result storage (optional)
            metric_mode: "fast" for 30 metrics, "full" for 117 metrics
        """
        self.chunk_size = chunk_size
        self.max_concurrent_http = max_concurrent_http
        self.requests_per_second = requests_per_second
        self.metric_mode = metric_mode

        self.async_client = AsyncDataClient(
            max_concurrent=max_concurrent_http,
            requests_per_second=requests_per_second,
        )

        # Optional Redis cache
        self._cache = None
        if redis_url:
            try:
                from src.cache.redis_cache import MetricCache
                self._cache = MetricCache(redis_url=redis_url)
                logger.info(f"Streaming scanner using Redis at {redis_url}")
            except Exception as e:
                logger.warning(f"Redis init failed: {e}. Results will use callback only.")

    async def scan_streaming_async(
        self,
        addresses: List[str],
        result_callback: Optional[callable] = None,
        progress_callback: Optional[callable] = None,
        max_trades_per_wallet: int = 1000,
    ) -> StreamingResult:
        """
        Scan wallets in a memory-safe streaming fashion.

        Args:
            addresses: List of wallet addresses (can be millions)
            result_callback: Called with (address, metrics, quant_score) for each wallet
            progress_callback: Called with (processed, total) for progress updates
            max_trades_per_wallet: Max trades to fetch per wallet

        Returns:
            StreamingResult with summary stats (not the actual metrics)
        """
        import gc

        result = StreamingResult(total_wallets=len(addresses))
        start_time = time.monotonic()

        # Process in chunks
        total_chunks = (len(addresses) + self.chunk_size - 1) // self.chunk_size

        for chunk_idx in range(total_chunks):
            chunk_start = chunk_idx * self.chunk_size
            chunk_end = min(chunk_start + self.chunk_size, len(addresses))
            chunk_addresses = addresses[chunk_start:chunk_end]

            # Fetch chunk
            activities = await self.async_client.get_multiple_wallet_activities(
                chunk_addresses,
                include_positions=False,
                max_trades_per_wallet=max_trades_per_wallet,
            )

            # Process chunk
            use_full = self.metric_mode == "full"

            for address, activity in activities.items():
                if not activity.trades:
                    result.wallets_no_trades += 1
                    continue

                try:
                    # Convert to FastWalletData
                    trades_data = [
                        {
                            'price': t.price,
                            'size': t.size,
                            'fee': t.fee,
                            'side': t.side,
                            'timestamp': t.timestamp,
                            'market_id': t.market_id or t.condition_id,
                            'value': t.value,
                        }
                        for t in activity.trades
                    ]

                    if use_full:
                        data = FastWalletData117.from_trades(address, trades_data)
                        metrics = calculate_117_metrics(data)
                        quant_score = calculate_quant_score(metrics)
                    else:
                        data = FastWalletData.from_trades(address, trades_data)
                        metrics = calculate_fast_metrics(data)
                        quant_score = 0.0

                    # Store result (don't keep in memory)
                    if self._cache:
                        self._cache.set_wallet_metrics(address, metrics)
                        if use_full:
                            classification = "bot" if quant_score >= 80 else "quant" if quant_score >= 50 else "human"
                            self._cache.set_wallet_classification(address, classification, quant_score)
                        result.results_stored_to = "redis"

                    if result_callback:
                        result_callback(address, metrics, quant_score)
                        result.results_stored_to = "callback"

                    result.wallets_processed += 1

                except Exception as e:
                    logger.debug(f"Error processing {address}: {e}")
                    result.wallets_failed += 1

            result.chunks_processed += 1

            # Progress update
            if progress_callback:
                progress_callback(result.wallets_processed + result.wallets_no_trades, len(addresses))

            # Clear memory after each chunk
            del activities
            gc.collect()

            logger.debug(f"Chunk {chunk_idx + 1}/{total_chunks} complete")

        result.total_time = time.monotonic() - start_time

        logger.info(
            f"Streaming scan complete: {result.wallets_processed:,} wallets in {result.total_time:.1f}s "
            f"({result.wallets_processed / result.total_time:,.0f} wallets/sec)"
        )

        return result

    def scan_streaming(
        self,
        addresses: List[str],
        result_callback: Optional[callable] = None,
        progress_callback: Optional[callable] = None,
        max_trades_per_wallet: int = 1000,
    ) -> StreamingResult:
        """Synchronous wrapper for scan_streaming_async."""
        return asyncio.run(
            self.scan_streaming_async(addresses, result_callback, progress_callback, max_trades_per_wallet)
        )


def stream_scan_to_redis(
    addresses: List[str],
    redis_url: str = "redis://localhost:6379",
    chunk_size: int = 1000,
    progress_callback: Optional[callable] = None,
) -> StreamingResult:
    """
    Scan unlimited wallets and store results in Redis.

    Memory-safe: processes in chunks, stores to Redis, clears RAM.

    Args:
        addresses: List of wallet addresses (can be millions)
        redis_url: Redis connection URL
        chunk_size: Wallets per chunk (default: 1000)
        progress_callback: Called with (processed, total)

    Returns:
        StreamingResult with summary (actual metrics in Redis)
    """
    scanner = StreamingScanner(
        chunk_size=chunk_size,
        redis_url=redis_url,
        metric_mode="full",
    )
    return scanner.scan_streaming(addresses, progress_callback=progress_callback)


def stream_scan_with_callback(
    addresses: List[str],
    result_callback: callable,
    chunk_size: int = 1000,
    progress_callback: Optional[callable] = None,
) -> StreamingResult:
    """
    Scan unlimited wallets and stream results to a callback.

    Memory-safe: processes in chunks, calls callback, clears RAM.

    Args:
        addresses: List of wallet addresses (can be millions)
        result_callback: Called with (address, metrics_dict, quant_score) for each wallet
        chunk_size: Wallets per chunk (default: 1000)
        progress_callback: Called with (processed, total)

    Returns:
        StreamingResult with summary

    Example:
        def save_to_db(address, metrics, score):
            db.insert(address=address, metrics=metrics, quant_score=score)

        result = stream_scan_with_callback(addresses, save_to_db)
    """
    scanner = StreamingScanner(
        chunk_size=chunk_size,
        metric_mode="full",
    )
    return scanner.scan_streaming(addresses, result_callback=result_callback, progress_callback=progress_callback)
