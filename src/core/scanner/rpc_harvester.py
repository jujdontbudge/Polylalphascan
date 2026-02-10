"""
RPC Parallel Harvester - High-throughput blockchain scanning for Polymarket wallets.

Bypasses API rate limits by querying OrderFilled events directly from Polygon RPC.
Target: 1000+ markets/second throughput using free RPC providers.
"""
import asyncio
import logging
import time
from typing import Set, List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from src.api.polygon_rpc_client import (
    PolygonRPCClient,
    ProviderPool,
    RPCProvider,
    OrderFilledEvent,
    CTF_EXCHANGE_MULTI,
    CTF_EXCHANGE_LEGACY,
    ORDER_FILLED_TOPIC,
    DEFAULT_PROVIDERS,
)

logger = logging.getLogger(__name__)

# Polygon block time is ~2 seconds
POLYGON_BLOCK_TIME = 2.0
BLOCKS_PER_HOUR = int(3600 / POLYGON_BLOCK_TIME)  # ~1800 blocks/hour
BLOCKS_PER_DAY = BLOCKS_PER_HOUR * 24  # ~43,200 blocks/day

# Polymarket CTF Exchange deployment block (approximate)
POLYMARKET_START_BLOCK = 30_000_000


@dataclass
class RPCHarvestResult:
    """Results from RPC-based wallet harvesting"""
    wallets: Set[str] = field(default_factory=set)
    events_processed: int = 0
    blocks_scanned: int = 0
    chunks_completed: int = 0
    chunks_failed: int = 0
    duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)
    start_block: int = 0
    end_block: int = 0
    _completed_chunks: Set[tuple] = field(default_factory=set)

    @property
    def total_wallets(self) -> int:
        return len(self.wallets)

    @property
    def blocks_per_second(self) -> float:
        if self.duration_seconds > 0:
            return self.blocks_scanned / self.duration_seconds
        return 0.0

    @property
    def wallets_per_second(self) -> float:
        if self.duration_seconds > 0:
            return self.total_wallets / self.duration_seconds
        return 0.0

    @property
    def success_rate(self) -> float:
        total = self.chunks_completed + self.chunks_failed
        if total > 0:
            return self.chunks_completed / total * 100
        return 0.0


@dataclass
class BlockChunk:
    """A chunk of blocks to process"""
    from_block: int
    to_block: int
    provider: Optional[RPCProvider] = None
    retry_count: int = 0
    max_retries: int = 3

    @property
    def size(self) -> int:
        return self.to_block - self.from_block + 1

    def can_retry(self) -> bool:
        return self.retry_count < self.max_retries


class RPCParallelHarvester:
    """
    High-throughput parallel harvester using direct RPC queries.

    Splits block ranges across multiple providers and processes in parallel.
    Target: ~375,000 blocks/second with 5 providers.
    """

    def __init__(
        self,
        providers: List[RPCProvider] = None,
        max_concurrent: int = 100,
        contracts: List[str] = None,
    ):
        """
        Initialize the RPC harvester.

        Args:
            providers: List of RPC providers (defaults to free providers)
            max_concurrent: Maximum concurrent RPC requests
            contracts: Contract addresses to monitor (defaults to both CTF exchanges)
        """
        self.providers = providers or [RPCProvider(**p.__dict__) for p in DEFAULT_PROVIDERS]
        self.pool = ProviderPool(self.providers)
        self.max_concurrent = max_concurrent
        self.contracts = contracts or [CTF_EXCHANGE_MULTI, CTF_EXCHANGE_LEGACY]

        # Stats
        self._total_requests = 0
        self._total_events = 0

    def _create_chunks(
        self,
        from_block: int,
        to_block: int,
    ) -> List[BlockChunk]:
        """
        Split a block range into optimal chunks based on provider capabilities.

        Uses a conservative chunk size that most providers can handle.
        """
        # Use 500 blocks per chunk as safe default for free RPC providers
        # Some providers like 1RPC limit to 500 blocks max
        chunk_size = 500

        chunks = []
        current = from_block

        while current <= to_block:
            chunk_end = min(current + chunk_size - 1, to_block)
            chunks.append(BlockChunk(from_block=current, to_block=chunk_end))
            current = chunk_end + 1

        return chunks

    async def _process_chunk(
        self,
        chunk: BlockChunk,
        client: PolygonRPCClient,
        semaphore: asyncio.Semaphore,
        result: RPCHarvestResult,
        result_lock: asyncio.Lock,
    ) -> bool:
        """
        Process a single block chunk.

        Returns True on success, False on failure.
        """
        async with semaphore:
            try:
                # Query events from both contracts
                all_wallets: Set[str] = set()
                total_events = 0

                for contract in self.contracts:
                    # Query logs directly (will use pool internally)
                    logs = await client.query_logs(
                        from_block=chunk.from_block,
                        to_block=chunk.to_block,
                        contract_address=contract,
                        topics=[ORDER_FILLED_TOPIC],
                    )

                    if logs is None:
                        # Query failed, mark for retry
                        chunk.retry_count += 1
                        return False

                    for log in logs:
                        event = client.parse_order_filled(log)
                        if event:
                            all_wallets.update(event.wallets)
                            total_events += 1

                # Update results (prevent double-counting on retries)
                async with result_lock:
                    result.wallets.update(all_wallets)
                    result.events_processed += total_events
                    chunk_key = (chunk.from_block, chunk.to_block)
                    if chunk_key not in result._completed_chunks:
                        result.blocks_scanned += chunk.size
                        result._completed_chunks.add(chunk_key)
                    result.chunks_completed += 1

                self._total_requests += 1
                self._total_events += total_events
                return True

            except Exception as e:
                chunk.retry_count += 1
                logger.warning(f"Chunk {chunk.from_block}-{chunk.to_block} failed: {e}")

                async with result_lock:
                    if not chunk.can_retry():
                        result.chunks_failed += 1
                        result.errors.append(f"Blocks {chunk.from_block}-{chunk.to_block}: {e}")

                return False

    async def harvest_block_range(
        self,
        from_block: int,
        to_block: int,
        progress_callback: Callable[[int, int, int], None] = None,
    ) -> RPCHarvestResult:
        """
        Harvest all wallets from a block range.

        Args:
            from_block: Start block (inclusive)
            to_block: End block (inclusive)
            progress_callback: Optional callback(blocks_done, total_blocks, wallets_found)

        Returns:
            RPCHarvestResult with all discovered wallets
        """
        result = RPCHarvestResult(start_block=from_block, end_block=to_block)
        start_time = time.monotonic()

        # Create chunks
        chunks = self._create_chunks(from_block, to_block)
        total_blocks = to_block - from_block + 1

        logger.info(
            f"Harvesting blocks {from_block:,} to {to_block:,} ({total_blocks:,} blocks in {len(chunks)} chunks)"
        )
        logger.info(f"Using {len(self.providers)} providers, {self.max_concurrent} concurrent requests")

        # Processing infrastructure
        semaphore = asyncio.Semaphore(self.max_concurrent)
        result_lock = asyncio.Lock()
        pending_chunks = list(chunks)
        retry_queue: List[BlockChunk] = []

        async with PolygonRPCClient(provider_pool=self.pool) as client:
            # Process all chunks with retries
            while pending_chunks or retry_queue:
                # Move retryable chunks back to pending
                for chunk in retry_queue:
                    if chunk.can_retry():
                        pending_chunks.append(chunk)
                retry_queue.clear()

                if not pending_chunks:
                    break

                # Process current batch
                tasks = [
                    self._process_chunk(chunk, client, semaphore, result, result_lock)
                    for chunk in pending_chunks
                ]

                completed = 0
                for coro in asyncio.as_completed(tasks):
                    success = await coro
                    completed += 1

                    # Check for failed chunks that can retry
                    for chunk in pending_chunks:
                        if chunk.retry_count > 0 and chunk.can_retry():
                            if chunk not in retry_queue:
                                retry_queue.append(chunk)

                    # Progress callback
                    if progress_callback and completed % 10 == 0:
                        progress_callback(
                            result.blocks_scanned,
                            total_blocks,
                            result.total_wallets,
                        )

                # Clear processed chunks
                pending_chunks.clear()

        result.duration_seconds = time.monotonic() - start_time

        logger.info(
            f"Harvest complete: {result.blocks_scanned:,} blocks, "
            f"{result.total_wallets:,} wallets, {result.events_processed:,} events "
            f"in {result.duration_seconds:.1f}s ({result.blocks_per_second:,.0f} blocks/s)"
        )

        return result

    async def harvest_time_range(
        self,
        hours: int = None,
        days: int = None,
        start_time: datetime = None,
        end_time: datetime = None,
        progress_callback: Callable[[int, int, int], None] = None,
    ) -> RPCHarvestResult:
        """
        Harvest wallets from a time range.

        Args:
            hours: Number of hours to look back
            days: Number of days to look back
            start_time: Explicit start datetime
            end_time: Explicit end datetime (defaults to now)
            progress_callback: Progress callback function

        Returns:
            RPCHarvestResult with all discovered wallets
        """
        # Get current block
        async with PolygonRPCClient(provider_pool=self.pool) as client:
            latest_block = await client.get_latest_block()
            if latest_block is None:
                logger.error("Failed to get latest block")
                return RPCHarvestResult()

        # Calculate block range
        if hours:
            blocks_back = hours * BLOCKS_PER_HOUR
        elif days:
            blocks_back = days * BLOCKS_PER_DAY
        elif start_time and end_time:
            duration = (end_time - start_time).total_seconds()
            blocks_back = int(duration / POLYGON_BLOCK_TIME)
            latest_block = latest_block - int(
                (datetime.utcnow() - end_time).total_seconds() / POLYGON_BLOCK_TIME
            )
        else:
            # Default to last 24 hours
            blocks_back = BLOCKS_PER_DAY

        from_block = max(POLYMARKET_START_BLOCK, latest_block - blocks_back)
        to_block = latest_block

        logger.info(f"Time range -> blocks {from_block:,} to {to_block:,}")

        return await self.harvest_block_range(from_block, to_block, progress_callback)

    async def harvest_all_history(
        self,
        from_block: int = POLYMARKET_START_BLOCK,
        progress_callback: Callable[[int, int, int], None] = None,
    ) -> RPCHarvestResult:
        """
        Harvest all historical wallets since Polymarket deployment.

        Warning: This will take a while (estimate: ~20 minutes for 30M blocks)

        Args:
            from_block: Starting block (default: Polymarket deployment)
            progress_callback: Progress callback function

        Returns:
            RPCHarvestResult with all historical wallets
        """
        async with PolygonRPCClient(provider_pool=self.pool) as client:
            latest_block = await client.get_latest_block()
            if latest_block is None:
                logger.error("Failed to get latest block")
                return RPCHarvestResult()

        total_blocks = latest_block - from_block
        logger.info(
            f"Full history harvest: {total_blocks:,} blocks from {from_block:,} to {latest_block:,}"
        )
        logger.info(f"Estimated time: ~{total_blocks / 25000:.0f} seconds")

        return await self.harvest_block_range(from_block, latest_block, progress_callback)

    def harvest_block_range_sync(
        self,
        from_block: int,
        to_block: int,
        progress_callback: Callable[[int, int, int], None] = None,
    ) -> RPCHarvestResult:
        """Synchronous wrapper for harvest_block_range"""
        return asyncio.run(self.harvest_block_range(from_block, to_block, progress_callback))

    def harvest_time_range_sync(
        self,
        hours: int = None,
        days: int = None,
        progress_callback: Callable[[int, int, int], None] = None,
    ) -> RPCHarvestResult:
        """Synchronous wrapper for harvest_time_range"""
        return asyncio.run(self.harvest_time_range(hours=hours, days=days, progress_callback=progress_callback))

    def harvest_all_history_sync(
        self,
        from_block: int = POLYMARKET_START_BLOCK,
        progress_callback: Callable[[int, int, int], None] = None,
    ) -> RPCHarvestResult:
        """Synchronous wrapper for harvest_all_history"""
        return asyncio.run(self.harvest_all_history(from_block, progress_callback))

    def get_stats(self) -> Dict[str, Any]:
        """Get harvester statistics"""
        return {
            "total_requests": self._total_requests,
            "total_events": self._total_events,
            "providers": self.pool.get_stats(),
        }


# Convenience functions
def harvest_recent_wallets(hours: int = 24) -> RPCHarvestResult:
    """
    Convenience function to harvest wallets from recent hours.

    Args:
        hours: Number of hours to look back (default: 24)

    Returns:
        RPCHarvestResult with discovered wallets
    """
    harvester = RPCParallelHarvester()
    return harvester.harvest_time_range_sync(hours=hours)


def harvest_blocks(from_block: int, to_block: int) -> RPCHarvestResult:
    """
    Convenience function to harvest wallets from a block range.

    Args:
        from_block: Start block
        to_block: End block

    Returns:
        RPCHarvestResult with discovered wallets
    """
    harvester = RPCParallelHarvester()
    return harvester.harvest_block_range_sync(from_block, to_block)
