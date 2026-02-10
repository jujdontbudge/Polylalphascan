"""
Async Data API Client - High-throughput async HTTP client for Polymarket APIs.

Designed for 1000+ wallets/second throughput with:
- aiohttp for async HTTP
- Global rate limiting with token bucket
- Connection pooling
- Automatic retries with exponential backoff
"""
import asyncio
import aiohttp
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from config.settings import api_config

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Individual trade data"""
    id: str
    transaction_hash: str
    timestamp: datetime
    market_id: str
    condition_id: str
    wallet_address: str
    side: str  # 'BUY' or 'SELL'
    outcome: str
    price: float
    size: float
    value: float
    fee: float
    asset_id: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trade":
        """Create Trade from API response dict"""
        timestamp_str = data.get("timestamp", data.get("createdAt", ""))
        if isinstance(timestamp_str, str) and timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except ValueError:
                timestamp = datetime.now()
        else:
            timestamp = datetime.now()

        wallet = (
            data.get("proxyWallet") or
            data.get("maker") or
            data.get("user") or
            data.get("wallet_address") or
            data.get("taker") or
            ""
        )

        size = float(data.get("size", data.get("amount", 0)) or 0)
        price = float(data.get("price", 0) or 0)
        value = float(data.get("value", data.get("usdcSize", 0)) or 0)
        if value == 0 and size > 0 and price > 0:
            value = size * price

        return cls(
            id=data.get("id", data.get("tradeId", "")),
            transaction_hash=data.get("transactionHash", data.get("transaction_hash", "")),
            timestamp=timestamp,
            market_id=data.get("marketId", data.get("market_id", data.get("market", ""))),
            condition_id=data.get("conditionId", data.get("condition_id", "")),
            wallet_address=wallet,
            side=data.get("side", data.get("type", "BUY")).upper(),
            outcome=data.get("outcome", data.get("outcomeName", data.get("title", ""))),
            price=price,
            size=size,
            value=value,
            fee=float(data.get("fee", data.get("feeAmount", 0)) or 0),
            asset_id=data.get("assetId", data.get("asset_id", data.get("asset", ""))),
        )


@dataclass
class Position:
    """Wallet position data"""
    market_id: str
    condition_id: str
    wallet_address: str
    outcome: str
    outcome_index: int
    size: float
    avg_price: float
    current_price: float
    value: float
    unrealized_pnl: float
    realized_pnl: float
    asset_id: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Position":
        """Create Position from API response dict"""
        size = float(data.get("size", data.get("amount", 0)) or 0)
        avg_price = float(data.get("avgPrice", data.get("averagePrice", 0)) or 0)
        current_price = float(data.get("currentPrice", data.get("price", 0)) or 0)
        value = size * current_price
        unrealized_pnl = (current_price - avg_price) * size

        return cls(
            market_id=data.get("marketId", data.get("market_id", "")),
            condition_id=data.get("conditionId", data.get("condition_id", "")),
            wallet_address=data.get("user", data.get("wallet_address", "")),
            outcome=data.get("outcome", data.get("outcomeName", "")),
            outcome_index=int(data.get("outcomeIndex", data.get("outcome_index", 0)) or 0),
            size=size,
            avg_price=avg_price,
            current_price=current_price,
            value=value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=float(data.get("realizedPnl", data.get("realized_pnl", 0)) or 0),
            asset_id=data.get("assetId", data.get("asset_id")),
        )


@dataclass
class WalletActivity:
    """Summary of wallet activity"""
    address: str
    trades: List[Trade] = field(default_factory=list)
    positions: List[Position] = field(default_factory=list)
    total_volume: float = 0.0
    total_trades: int = 0
    total_pnl: float = 0.0
    first_trade: Optional[datetime] = None
    last_trade: Optional[datetime] = None

    def calculate_summary(self):
        """Calculate summary statistics from trades"""
        if self.trades:
            self.total_trades = len(self.trades)
            self.total_volume = sum(t.value for t in self.trades)
            self.first_trade = min(t.timestamp for t in self.trades)
            self.last_trade = max(t.timestamp for t in self.trades)

        if self.positions:
            self.total_pnl = sum(p.realized_pnl + p.unrealized_pnl for p in self.positions)


class AsyncRateLimiter:
    """
    Token bucket rate limiter for async operations.

    Coordinates rate limiting across all concurrent workers.
    """

    def __init__(
        self,
        requests_per_second: float = 50.0,
        burst_size: int = 100,
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
        """Acquire a token, waiting if necessary. Returns wait time."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.last_update = now

            # Refill tokens
            self.tokens = min(
                self.burst_size,
                self.tokens + elapsed * self.rate
            )

            wait_time = 0.0
            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1

            self.total_requests += 1
            self.total_wait_time += wait_time
            return wait_time


class AsyncDataClient:
    """
    High-throughput async HTTP client for Polymarket Data API.

    Designed for 1000+ wallets/second with:
    - aiohttp connection pooling
    - Global rate limiting
    - Automatic retries
    """

    def __init__(
        self,
        max_concurrent: int = 200,
        requests_per_second: float = 100.0,
        max_retries: int = 3,
        timeout_seconds: int = 30,
    ):
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)

        self.rate_limiter = AsyncRateLimiter(
            requests_per_second=requests_per_second,
            burst_size=min(200, int(requests_per_second * 2)),
        )

        self.base_url = api_config.DATA_API_BASE.rstrip("/")

        # Stats
        self.successful_requests = 0
        self.failed_requests = 0

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
                        self.successful_requests += 1
                        return await response.json()
                    elif response.status == 429:
                        wait = 2 ** attempt
                        logger.warning(f"Rate limited, waiting {wait}s...")
                        await asyncio.sleep(wait)
                    else:
                        logger.debug(f"HTTP {response.status} for {url}")
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(0.5)
            except asyncio.TimeoutError:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(0.5)
            except Exception as e:
                logger.debug(f"Error fetching {url}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(0.5)

        self.failed_requests += 1
        return None

    async def _fetch_paginated(
        self,
        session: aiohttp.ClientSession,
        endpoint: str,
        params: Dict[str, Any] = None,
        page_size: int = 500,
        max_pages: Optional[int] = None,
    ) -> List[Any]:
        """Fetch all pages of a paginated endpoint"""
        all_results = []
        params = params or {}
        params["limit"] = page_size
        offset = 0
        page_count = 0

        url = f"{self.base_url}/{endpoint}"

        while True:
            params["offset"] = offset
            response = await self._fetch_json(session, url, params)

            if response is None:
                break

            items = response if isinstance(response, list) else response.get("data", [])

            if not items:
                break

            all_results.extend(items)
            page_count += 1

            if max_pages and page_count >= max_pages:
                break

            if len(items) < page_size:
                break

            offset += page_size

        return all_results

    async def get_wallet_activity(
        self,
        session: aiohttp.ClientSession,
        wallet_address: str,
        include_positions: bool = True,
        max_trades: Optional[int] = None,
    ) -> WalletActivity:
        """
        Get full activity for a wallet asynchronously.

        Uses asyncio.gather() to fetch trades and positions in parallel
        for ~1.5x speedup on fetch phase.

        Args:
            session: aiohttp session
            wallet_address: Wallet address
            include_positions: Whether to fetch positions
            max_trades: Maximum trades to fetch

        Returns:
            WalletActivity with trades and positions
        """
        activity = WalletActivity(address=wallet_address)

        max_pages = None
        if max_trades:
            max_pages = (max_trades // 500) + 1

        # Fetch trades and positions in parallel using asyncio.gather()
        trades_task = self._fetch_paginated(
            session,
            "activity",
            params={"user": wallet_address, "type": "TRADE"},
            max_pages=max_pages,
        )

        if include_positions:
            positions_url = f"{self.base_url}/positions"
            positions_task = self._fetch_json(
                session, positions_url, params={"user": wallet_address}
            )
            # Parallel fetch - trades and positions simultaneously
            trades_data, positions_data = await asyncio.gather(
                trades_task, positions_task
            )
        else:
            trades_data = await trades_task
            positions_data = None

        # Process trades
        activity.trades = [Trade.from_dict(t) for t in trades_data]

        # Process positions
        if positions_data:
            positions_list = (
                positions_data if isinstance(positions_data, list)
                else positions_data.get("positions", [])
            )
            activity.positions = [Position.from_dict(p) for p in positions_list]

        activity.calculate_summary()
        return activity

    async def get_multiple_wallet_activities(
        self,
        wallet_addresses: List[str],
        include_positions: bool = True,
        max_trades_per_wallet: Optional[int] = None,
        progress_callback: callable = None,
    ) -> Dict[str, WalletActivity]:
        """
        Fetch activity for multiple wallets in parallel.

        Args:
            wallet_addresses: List of wallet addresses
            include_positions: Whether to fetch positions
            max_trades_per_wallet: Max trades per wallet
            progress_callback: Optional callback(completed, total)

        Returns:
            Dict of address -> WalletActivity
        """
        results = {}
        semaphore = asyncio.Semaphore(self.max_concurrent)

        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent,
            limit_per_host=self.max_concurrent,
        )

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
            "User-Agent": "PolyScan/3.0",
        }

        async def fetch_one(address: str) -> Tuple[str, Optional[WalletActivity]]:
            async with semaphore:
                try:
                    activity = await self.get_wallet_activity(
                        session, address, include_positions, max_trades_per_wallet
                    )
                    return address, activity
                except Exception as e:
                    logger.debug(f"Error fetching {address}: {e}")
                    return address, None

        async with aiohttp.ClientSession(
            timeout=self.timeout,
            connector=connector,
            headers=headers,
        ) as session:
            tasks = [fetch_one(addr) for addr in wallet_addresses]

            completed = 0
            for coro in asyncio.as_completed(tasks):
                address, activity = await coro
                if activity and activity.trades:
                    results[address] = activity

                completed += 1
                if progress_callback and completed % 50 == 0:
                    progress_callback(completed, len(wallet_addresses))

        return results

    def get_wallet_activity_sync(
        self,
        wallet_address: str,
        include_positions: bool = True,
        max_trades: Optional[int] = None,
    ) -> WalletActivity:
        """Synchronous wrapper for get_wallet_activity"""
        async def _fetch():
            connector = aiohttp.TCPConnector(limit=10)
            async with aiohttp.ClientSession(
                timeout=self.timeout,
                connector=connector,
            ) as session:
                return await self.get_wallet_activity(
                    session, wallet_address, include_positions, max_trades
                )
        return asyncio.run(_fetch())

    def get_multiple_wallet_activities_sync(
        self,
        wallet_addresses: List[str],
        include_positions: bool = True,
        max_trades_per_wallet: Optional[int] = None,
        progress_callback: callable = None,
    ) -> Dict[str, WalletActivity]:
        """Synchronous wrapper for get_multiple_wallet_activities"""
        return asyncio.run(
            self.get_multiple_wallet_activities(
                wallet_addresses,
                include_positions,
                max_trades_per_wallet,
                progress_callback,
            )
        )


# Convenience function
async def fetch_wallets_parallel(
    wallet_addresses: List[str],
    max_concurrent: int = 200,
    requests_per_second: float = 100.0,
) -> Dict[str, WalletActivity]:
    """
    Convenience function to fetch multiple wallets in parallel.

    Args:
        wallet_addresses: List of wallet addresses
        max_concurrent: Maximum concurrent requests
        requests_per_second: Rate limit

    Returns:
        Dict of address -> WalletActivity
    """
    client = AsyncDataClient(
        max_concurrent=max_concurrent,
        requests_per_second=requests_per_second,
    )
    return await client.get_multiple_wallet_activities(wallet_addresses)
