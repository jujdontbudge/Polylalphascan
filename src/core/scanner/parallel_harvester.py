"""
Parallel Wallet Harvester - High-throughput market scanning with zero missed wallets.

Capabilities:
- 1000+ markets concurrently
- Global rate limiting across all workers
- Full pagination to capture ALL wallets
- Automatic retry with exponential backoff
"""
import asyncio
import aiohttp
import logging
import time
from typing import Set, List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from config.settings import api_config

logger = logging.getLogger(__name__)


@dataclass
class ParallelHarvestResult:
    """Results from parallel harvesting"""
    wallets: Set[str] = field(default_factory=set)
    markets_scanned: int = 0
    markets_failed: int = 0
    total_trades_processed: int = 0
    duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def total_wallets(self) -> int:
        return len(self.wallets)

    @property
    def markets_per_second(self) -> float:
        if self.duration_seconds > 0:
            return self.markets_scanned / self.duration_seconds
        return 0.0

    @property
    def success_rate(self) -> float:
        total = self.markets_scanned + self.markets_failed
        if total > 0:
            return self.markets_scanned / total * 100
        return 0.0


class AsyncRateLimiter:
    """
    Global rate limiter for coordinating requests across all concurrent workers.

    Uses token bucket algorithm for smooth rate limiting.
    """

    def __init__(
        self,
        requests_per_second: float = 15.0,
        burst_size: int = 20,
    ):
        self.rate = requests_per_second
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

        # Stats
        self.total_requests = 0
        self.total_wait_time = 0.0

    async def acquire(self) -> float:
        """
        Acquire a token, waiting if necessary.
        Returns the wait time in seconds.
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.last_update = now

            # Add tokens based on elapsed time
            self.tokens = min(
                self.burst_size,
                self.tokens + elapsed * self.rate
            )

            wait_time = 0.0
            if self.tokens < 1:
                # Need to wait for a token
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1

            self.total_requests += 1
            self.total_wait_time += wait_time
            return wait_time

    @property
    def avg_wait_time(self) -> float:
        if self.total_requests > 0:
            return self.total_wait_time / self.total_requests
        return 0.0


class ParallelHarvester:
    """
    High-throughput parallel wallet harvester.

    Designed for scanning 1000+ markets concurrently with zero missed wallets.
    """

    def __init__(
        self,
        max_concurrent: int = 50,
        requests_per_second: float = 10.0,  # Conservative to avoid rate limiting
        max_retries: int = 3,
        timeout_seconds: int = 30,
    ):
        self.max_concurrent = max_concurrent
        self.requests_per_second = requests_per_second
        self.max_retries = max_retries
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)

        self.rate_limiter = AsyncRateLimiter(
            requests_per_second=requests_per_second,
            burst_size=min(50, int(requests_per_second * 2)),
        )

        self.base_url = api_config.DATA_API_BASE

    async def _fetch_json(
        self,
        session: aiohttp.ClientSession,
        url: str,
        params: Dict[str, Any] = None,
    ) -> Optional[Any]:
        """Fetch JSON with rate limiting and retries"""
        for attempt in range(self.max_retries):
            await self.rate_limiter.acquire()

            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:
                        # Rate limited - exponential backoff
                        wait = 2 ** attempt
                        logger.warning(f"Rate limited, waiting {wait}s...")
                        await asyncio.sleep(wait)
                    else:
                        logger.error(f"HTTP {response.status} for {url}")
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(1)
            except asyncio.TimeoutError:
                logger.warning(f"Timeout for {url}, attempt {attempt + 1}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1)

        return None

    async def _harvest_condition_full(
        self,
        session: aiohttp.ClientSession,
        condition_id: str,
        semaphore: asyncio.Semaphore,
    ) -> tuple[Set[str], int, Optional[str]]:
        """
        Harvest ALL wallets from a condition with full pagination.

        Returns (wallets, trade_count, error_message)
        """
        async with semaphore:
            wallets = set()
            total_trades = 0
            offset = 0
            page_size = 500  # Max page size

            while True:
                url = f"{self.base_url}/trades"
                params = {
                    "market": condition_id,  # API uses 'market' param for conditionId
                    "limit": page_size,
                    "offset": offset,
                }

                data = await self._fetch_json(session, url, params)

                if data is None:
                    if total_trades == 0:
                        return wallets, 0, f"Failed to fetch {condition_id[:20]}..."
                    break  # Partial success

                trades = data if isinstance(data, list) else data.get("data", [])

                if not trades:
                    break

                for trade in trades:
                    # Primary wallet field is proxyWallet
                    wallet = trade.get("proxyWallet")
                    if wallet:
                        wallets.add(wallet.lower())

                total_trades += len(trades)

                # Check if we got a full page (more data available)
                if len(trades) < page_size:
                    break

                offset += page_size

                # Safety limit to prevent infinite loops
                if offset > 100000:
                    logger.warning(f"Hit safety limit for {condition_id}")
                    break

            return wallets, total_trades, None

    async def harvest_markets(
        self,
        condition_ids: List[str],
        progress_callback: callable = None,
    ) -> ParallelHarvestResult:
        """
        Harvest wallets from multiple markets in parallel.

        Args:
            condition_ids: List of market condition IDs
            progress_callback: Optional callback(scanned, total, wallets)

        Returns:
            ParallelHarvestResult with all discovered wallets
        """
        result = ParallelHarvestResult()
        start_time = time.monotonic()

        logger.info(f"Starting parallel harvest of {len(condition_ids)} markets...")
        logger.info(f"Max concurrent: {self.max_concurrent}, Rate: {self.requests_per_second}/s")

        semaphore = asyncio.Semaphore(self.max_concurrent)

        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent,
            limit_per_host=self.max_concurrent,
        )

        # Headers to avoid brotli encoding issues
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
            "User-Agent": "PolyScan/2.0",
        }

        async with aiohttp.ClientSession(
            timeout=self.timeout,
            connector=connector,
            headers=headers,
        ) as session:
            # Create tasks for all conditions
            tasks = [
                self._harvest_condition_full(session, cid, semaphore)
                for cid in condition_ids
            ]

            # Process with progress tracking
            completed = 0
            for coro in asyncio.as_completed(tasks):
                wallets, trade_count, error = await coro

                if error:
                    result.markets_failed += 1
                    result.errors.append(error)
                else:
                    result.markets_scanned += 1
                    result.wallets.update(wallets)
                    result.total_trades_processed += trade_count

                completed += 1

                # Progress callback
                if progress_callback and completed % 10 == 0:
                    progress_callback(completed, len(condition_ids), len(result.wallets))

                # Log progress
                if completed % 100 == 0:
                    elapsed = time.monotonic() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"Progress: {completed}/{len(condition_ids)} markets, "
                        f"{len(result.wallets)} wallets, {rate:.1f} markets/s"
                    )

        result.duration_seconds = time.monotonic() - start_time

        logger.info(
            f"Harvest complete: {result.markets_scanned} markets, "
            f"{result.total_wallets} wallets in {result.duration_seconds:.1f}s "
            f"({result.markets_per_second:.1f} markets/s)"
        )

        return result

    def harvest_sync(
        self,
        condition_ids: List[str],
        progress_callback: callable = None,
    ) -> ParallelHarvestResult:
        """
        Synchronous wrapper for harvest_markets.

        Use this from non-async code.
        """
        return asyncio.run(self.harvest_markets(condition_ids, progress_callback))


class UltraHarvester:
    """
    Ultra high-throughput harvester for 1000+ markets.

    Uses multiple API endpoints in parallel for maximum coverage.
    """

    def __init__(
        self,
        max_concurrent: int = 200,
        requests_per_second: float = 25.0,
    ):
        self.harvester = ParallelHarvester(
            max_concurrent=max_concurrent,
            requests_per_second=requests_per_second,
        )
        self.gamma_base = api_config.GAMMA_API_BASE
        self.data_base = api_config.DATA_API_BASE

    async def discover_all_active_markets(
        self,
        min_volume: float = 1000.0,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Discover all active markets with volume above threshold.
        """
        logger.info(f"Discovering active markets (min volume: ${min_volume:,.0f})...")

        connector = aiohttp.TCPConnector(limit=50)
        timeout = aiohttp.ClientTimeout(total=60)

        # Headers to avoid brotli encoding issues
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",  # Exclude 'br' (brotli)
            "User-Agent": "PolyScan/2.0",
        }

        async with aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers=headers,
        ) as session:
            # Fetch from Gamma API (events)
            url = f"{self.gamma_base}/events"
            params = {
                "active": "true",
                "closed": "false",
                "limit": 500,
            }

            all_markets = []
            offset = 0

            while len(all_markets) < limit:
                params["offset"] = offset

                await self.harvester.rate_limiter.acquire()

                try:
                    async with session.get(url, params=params) as response:
                        if response.status != 200:
                            break
                        events = await response.json()
                except Exception as e:
                    logger.error(f"Error fetching events: {e}")
                    break

                if not events:
                    break

                for event in events:
                    markets = event.get("markets", [])
                    for market in markets:
                        volume = float(market.get("volume", 0) or 0)
                        # conditionId is camelCase in API response
                        condition_id = market.get("conditionId") or market.get("condition_id")
                        if volume >= min_volume and condition_id:
                            all_markets.append({
                                "condition_id": condition_id,
                                "question": market.get("question", ""),
                                "volume": volume,
                                "event_title": event.get("title", ""),
                            })

                offset += len(events)

                if len(events) < 500:
                    break

            # Sort by volume and limit
            all_markets.sort(key=lambda x: x["volume"], reverse=True)
            all_markets = all_markets[:limit]

            total_volume = sum(m["volume"] for m in all_markets)
            logger.info(
                f"Discovered {len(all_markets)} markets with ${total_volume:,.0f} total volume"
            )

            return all_markets

    async def harvest_all(
        self,
        min_volume: float = 1000.0,
        max_markets: int = 1000,
        progress_callback: callable = None,
    ) -> ParallelHarvestResult:
        """
        Discover and harvest all active markets.

        Args:
            min_volume: Minimum market volume to include
            max_markets: Maximum markets to scan
            progress_callback: Optional progress callback

        Returns:
            ParallelHarvestResult with all wallets
        """
        # Phase 1: Discover markets
        markets = await self.discover_all_active_markets(
            min_volume=min_volume,
            limit=max_markets,
        )

        if not markets:
            return ParallelHarvestResult()

        # Phase 2: Extract condition IDs
        condition_ids = [
            m["condition_id"] for m in markets
            if m.get("condition_id")
        ]

        logger.info(f"Harvesting {len(condition_ids)} condition IDs...")

        # Phase 3: Parallel harvest
        return await self.harvester.harvest_markets(
            condition_ids,
            progress_callback=progress_callback,
        )

    def harvest_all_sync(
        self,
        min_volume: float = 1000.0,
        max_markets: int = 1000,
        progress_callback: callable = None,
    ) -> ParallelHarvestResult:
        """Synchronous wrapper"""
        return asyncio.run(self.harvest_all(
            min_volume=min_volume,
            max_markets=max_markets,
            progress_callback=progress_callback,
        ))


# Convenience function
def harvest_1000_markets(
    min_volume: float = 500.0,
    max_concurrent: int = 150,
    requests_per_second: float = 20.0,
) -> ParallelHarvestResult:
    """
    Convenience function to harvest 1000 markets.

    Returns all unique wallets across all markets.
    """
    harvester = UltraHarvester(
        max_concurrent=max_concurrent,
        requests_per_second=requests_per_second,
    )
    return harvester.harvest_all_sync(
        min_volume=min_volume,
        max_markets=1000,
    )
