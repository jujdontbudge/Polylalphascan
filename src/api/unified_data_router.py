"""
Unified Data Router - Smart data source routing for optimal performance.

Routes requests across multiple data sources:
1. Local trade index (instant if cached)
2. Goldsky subgraphs (batch-optimized)
3. REST API (fallback)
4. HyperSync (blockchain events)

Automatically selects the best source based on:
- Data freshness requirements
- Batch size optimization
- Source availability
"""
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from src.api.async_data_client import AsyncDataClient, Trade, Position, WalletActivity

logger = logging.getLogger(__name__)

# Try to import optional data sources
try:
    from src.api.goldsky_client import GoldskyClient, GoldskyTrade, GoldskyPosition
    GOLDSKY_AVAILABLE = True
except ImportError:
    GOLDSKY_AVAILABLE = False
    logger.debug("Goldsky client not available")

try:
    from src.api.hypersync_client import HyperSyncClient
    HYPERSYNC_AVAILABLE = True
except ImportError:
    HYPERSYNC_AVAILABLE = False
    logger.debug("HyperSync client not available")


@dataclass
class DataSourceStats:
    """Statistics for a data source"""
    name: str
    requests: int = 0
    successes: int = 0
    failures: int = 0
    total_latency_ms: float = 0.0
    last_used: Optional[datetime] = None

    @property
    def success_rate(self) -> float:
        if self.requests == 0:
            return 0.0
        return self.successes / self.requests

    @property
    def avg_latency_ms(self) -> float:
        if self.successes == 0:
            return 0.0
        return self.total_latency_ms / self.successes


@dataclass
class CachedWalletData:
    """Cached wallet data with freshness tracking"""
    trades: List[Trade]
    positions: List[Position]
    fetched_at: datetime
    source: str

    def is_fresh(self, max_age_seconds: int = 3600) -> bool:
        """Check if cache is fresh enough"""
        age = (datetime.utcnow() - self.fetched_at).total_seconds()
        return age < max_age_seconds


class UnifiedDataRouter:
    """
    Smart data router that selects optimal source for each request.

    Priority order (when all available):
    1. Local memory cache (instant)
    2. Goldsky subgraphs (batch queries, lowest latency for multi-wallet)
    3. REST API (reliable fallback)
    4. HyperSync (blockchain direct, for gap filling)

    Features:
    - Automatic failover between sources
    - Batch optimization for multi-wallet requests
    - Freshness-aware caching
    - Source health tracking
    """

    def __init__(
        self,
        cache_ttl_seconds: int = 3600,
        prefer_batch: bool = True,
        enable_goldsky: bool = True,
        enable_hypersync: bool = True,
    ):
        self.cache_ttl = cache_ttl_seconds
        self.prefer_batch = prefer_batch

        # Initialize data sources
        self.rest_client = AsyncDataClient()

        if enable_goldsky and GOLDSKY_AVAILABLE:
            self.goldsky_client = GoldskyClient()
        else:
            self.goldsky_client = None

        if enable_hypersync and HYPERSYNC_AVAILABLE:
            self.hypersync_client = HyperSyncClient()
        else:
            self.hypersync_client = None

        # In-memory cache
        self._cache: Dict[str, CachedWalletData] = {}

        # Source statistics
        self.stats = {
            "cache": DataSourceStats(name="cache"),
            "goldsky": DataSourceStats(name="goldsky"),
            "rest": DataSourceStats(name="rest"),
            "hypersync": DataSourceStats(name="hypersync"),
        }

    def _get_cached(self, address: str) -> Optional[CachedWalletData]:
        """Get cached data if fresh"""
        addr_lower = address.lower()
        cached = self._cache.get(addr_lower)
        if cached and cached.is_fresh(self.cache_ttl):
            self.stats["cache"].requests += 1
            self.stats["cache"].successes += 1
            return cached
        return None

    def _set_cache(
        self,
        address: str,
        trades: List[Trade],
        positions: List[Position],
        source: str,
    ):
        """Update cache for address"""
        self._cache[address.lower()] = CachedWalletData(
            trades=trades,
            positions=positions,
            fetched_at=datetime.utcnow(),
            source=source,
        )

    async def get_wallet_trades(
        self,
        wallet_address: str,
        include_positions: bool = True,
        force_refresh: bool = False,
    ) -> WalletActivity:
        """
        Get trades for a single wallet using optimal data source.

        Args:
            wallet_address: Wallet address
            include_positions: Include position data
            force_refresh: Skip cache

        Returns:
            WalletActivity with trades and positions
        """
        address = wallet_address.lower()

        # Check cache first
        if not force_refresh:
            cached = self._get_cached(address)
            if cached:
                activity = WalletActivity(address=address)
                activity.trades = cached.trades
                activity.positions = cached.positions
                activity.calculate_summary()
                return activity

        # Try Goldsky first (lower latency)
        if self.goldsky_client:
            try:
                start = time.monotonic()
                data = await self.goldsky_client.get_all_wallets_data([address])

                if address in data and data[address]["trades"]:
                    self.stats["goldsky"].requests += 1
                    self.stats["goldsky"].successes += 1
                    self.stats["goldsky"].total_latency_ms += (time.monotonic() - start) * 1000
                    self.stats["goldsky"].last_used = datetime.utcnow()

                    # Convert Goldsky data to standard format
                    trades = self._convert_goldsky_trades(data[address]["trades"])
                    positions = self._convert_goldsky_positions(data[address]["positions"])

                    self._set_cache(address, trades, positions, "goldsky")

                    activity = WalletActivity(address=address)
                    activity.trades = trades
                    activity.positions = positions
                    activity.calculate_summary()
                    return activity
            except Exception as e:
                self.stats["goldsky"].requests += 1
                self.stats["goldsky"].failures += 1
                logger.debug(f"Goldsky failed for {address}: {e}")

        # Fallback to REST API
        try:
            start = time.monotonic()
            import aiohttp

            connector = aiohttp.TCPConnector(limit=10)
            async with aiohttp.ClientSession(connector=connector) as session:
                activity = await self.rest_client.get_wallet_activity(
                    session, address, include_positions
                )

            self.stats["rest"].requests += 1
            self.stats["rest"].successes += 1
            self.stats["rest"].total_latency_ms += (time.monotonic() - start) * 1000
            self.stats["rest"].last_used = datetime.utcnow()

            self._set_cache(address, activity.trades, activity.positions, "rest")

            return activity

        except Exception as e:
            self.stats["rest"].requests += 1
            self.stats["rest"].failures += 1
            logger.error(f"REST API failed for {address}: {e}")

            # Return empty activity on complete failure
            return WalletActivity(address=address)

    async def get_multiple_wallets(
        self,
        wallet_addresses: List[str],
        include_positions: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, WalletActivity]:
        """
        Get data for multiple wallets using batch-optimized sources.

        For large batches, prefers Goldsky (100 wallets/request) over REST.

        Args:
            wallet_addresses: List of wallet addresses
            include_positions: Include position data
            progress_callback: Optional callback(completed, total)

        Returns:
            Dict mapping address -> WalletActivity
        """
        results: Dict[str, WalletActivity] = {}
        addresses = [addr.lower() for addr in wallet_addresses]

        # Check cache first
        uncached = []
        for addr in addresses:
            cached = self._get_cached(addr)
            if cached:
                activity = WalletActivity(address=addr)
                activity.trades = cached.trades
                activity.positions = cached.positions
                activity.calculate_summary()
                results[addr] = activity
            else:
                uncached.append(addr)

        if progress_callback:
            progress_callback(len(results), len(addresses))

        if not uncached:
            return results

        # Use Goldsky for batch fetch if available and batch is large enough
        if self.goldsky_client and len(uncached) >= 10 and self.prefer_batch:
            try:
                start = time.monotonic()
                goldsky_data = await self.goldsky_client.get_all_wallets_data(uncached)

                fetched = 0
                for addr, data in goldsky_data.items():
                    if data["trades"]:
                        trades = self._convert_goldsky_trades(data["trades"])
                        positions = self._convert_goldsky_positions(data["positions"])

                        self._set_cache(addr, trades, positions, "goldsky")

                        activity = WalletActivity(address=addr)
                        activity.trades = trades
                        activity.positions = positions
                        activity.calculate_summary()
                        results[addr] = activity

                        fetched += 1
                        if addr in uncached:
                            uncached.remove(addr)

                if fetched > 0:
                    self.stats["goldsky"].requests += 1
                    self.stats["goldsky"].successes += 1
                    self.stats["goldsky"].total_latency_ms += (time.monotonic() - start) * 1000
                    self.stats["goldsky"].last_used = datetime.utcnow()

                if progress_callback:
                    progress_callback(len(results), len(addresses))

            except Exception as e:
                self.stats["goldsky"].failures += 1
                logger.warning(f"Goldsky batch failed: {e}")

        # Fallback to REST for remaining
        if uncached:
            try:
                rest_results = await self.rest_client.get_multiple_wallet_activities(
                    uncached,
                    include_positions=include_positions,
                    progress_callback=lambda c, t: progress_callback(len(results) + c, len(addresses)) if progress_callback else None,
                )

                for addr, activity in rest_results.items():
                    self._set_cache(addr, activity.trades, activity.positions, "rest")
                    results[addr] = activity

                self.stats["rest"].requests += 1
                self.stats["rest"].successes += 1
                self.stats["rest"].last_used = datetime.utcnow()

            except Exception as e:
                self.stats["rest"].failures += 1
                logger.error(f"REST batch failed: {e}")

        return results

    async def discover_wallets(
        self,
        from_block: int = 30_000_000,
        max_wallets: int = 100_000,
    ) -> Set[str]:
        """
        Discover wallet addresses from blockchain events.

        Uses HyperSync if available, otherwise returns empty set.

        Args:
            from_block: Starting block
            max_wallets: Maximum wallets to return

        Returns:
            Set of wallet addresses
        """
        if not self.hypersync_client:
            logger.warning("HyperSync not available for wallet discovery")
            return set()

        try:
            start = time.monotonic()
            wallets = await self.hypersync_client.discover_wallets(from_block, max_wallets)

            self.stats["hypersync"].requests += 1
            self.stats["hypersync"].successes += 1
            self.stats["hypersync"].total_latency_ms += (time.monotonic() - start) * 1000
            self.stats["hypersync"].last_used = datetime.utcnow()

            return wallets

        except Exception as e:
            self.stats["hypersync"].failures += 1
            logger.error(f"HyperSync discovery failed: {e}")
            return set()

    def _convert_goldsky_trades(self, goldsky_trades: List[Any]) -> List[Trade]:
        """Convert Goldsky trades to standard Trade format"""
        trades = []
        for gt in goldsky_trades:
            trade = Trade(
                id=gt.id,
                transaction_hash=gt.transaction_hash,
                timestamp=gt.timestamp,
                market_id=gt.market_id,
                condition_id=gt.condition_id,
                wallet_address=gt.user,
                side=gt.side,
                outcome="",
                price=gt.price,
                size=gt.size,
                value=gt.value,
                fee=gt.fee,
            )
            trades.append(trade)
        return trades

    def _convert_goldsky_positions(self, goldsky_positions: List[Any]) -> List[Position]:
        """Convert Goldsky positions to standard Position format"""
        positions = []
        for gp in goldsky_positions:
            position = Position(
                market_id=gp.market_id,
                condition_id=gp.condition_id,
                wallet_address=gp.user,
                outcome=gp.outcome,
                outcome_index=gp.outcome_index,
                size=gp.size,
                avg_price=gp.avg_price,
                current_price=gp.current_price,
                value=gp.size * gp.current_price,
                unrealized_pnl=(gp.current_price - gp.avg_price) * gp.size,
                realized_pnl=gp.realized_pnl,
            )
            positions.append(position)
        return positions

    def get_stats_summary(self) -> Dict[str, Any]:
        """Get summary of data source statistics"""
        return {
            name: {
                "requests": stats.requests,
                "success_rate": f"{stats.success_rate:.1%}",
                "avg_latency_ms": f"{stats.avg_latency_ms:.1f}",
                "last_used": stats.last_used.isoformat() if stats.last_used else None,
            }
            for name, stats in self.stats.items()
        }

    def clear_cache(self):
        """Clear the in-memory cache"""
        self._cache.clear()

    def cache_size(self) -> int:
        """Get number of cached wallets"""
        return len(self._cache)


# Convenience function
async def fetch_wallets_unified(
    wallet_addresses: List[str],
    prefer_batch: bool = True,
) -> Dict[str, WalletActivity]:
    """
    Convenience function to fetch wallets using unified router.

    Args:
        wallet_addresses: List of wallet addresses
        prefer_batch: Prefer batch queries when possible

    Returns:
        Dict mapping address -> WalletActivity
    """
    router = UnifiedDataRouter(prefer_batch=prefer_batch)
    return await router.get_multiple_wallets(wallet_addresses)
