"""
Polygon RPC Client - Direct blockchain queries for Polymarket CTF Exchange events.

Bypasses API rate limits by querying OrderFilled events directly from Polygon RPC.
Supports multiple free RPC providers with automatic rotation and failover.
"""
import asyncio
import aiohttp
import logging
import time
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Polymarket CTF Exchange contract addresses on Polygon
CTF_EXCHANGE_MULTI = "0xC5d563A36AE78145C45a50134d48A1215220f80a"  # Multi-outcome
CTF_EXCHANGE_LEGACY = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"  # Legacy binary

# OrderFilled event signature (keccak256 hash)
# event OrderFilled(bytes32 indexed orderHash, address indexed maker, address indexed taker,
#                   uint256 makerAssetId, uint256 takerAssetId, uint256 makerAmountFilled,
#                   uint256 takerAmountFilled, uint256 fee)
# keccak256("OrderFilled(bytes32,address,address,uint256,uint256,uint256,uint256,uint256)")
ORDER_FILLED_TOPIC = "0xd0a08e8c493f9c94f29311604c9de1b4e8c8d4c06bd0c789af57f2d65bfec0f6"


class ProviderHealth(Enum):
    """Health status for RPC providers"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class RPCProvider:
    """Configuration for a single RPC provider"""
    name: str
    url: str
    rate_limit: float  # requests per second
    max_block_range: int  # max blocks per eth_getLogs query
    priority: int = 1  # higher = preferred

    # Runtime state (not part of config)
    health: ProviderHealth = field(default=ProviderHealth.HEALTHY, compare=False)
    consecutive_errors: int = field(default=0, compare=False)
    last_error_time: float = field(default=0.0, compare=False)
    total_requests: int = field(default=0, compare=False)
    total_errors: int = field(default=0, compare=False)
    last_request_time: float = field(default=0.0, compare=False)

    def mark_success(self):
        """Mark a successful request"""
        self.consecutive_errors = 0
        self.total_requests += 1
        if self.health != ProviderHealth.HEALTHY:
            self.health = ProviderHealth.HEALTHY
            logger.info(f"Provider {self.name} recovered to healthy")

    def mark_error(self):
        """Mark a failed request"""
        self.consecutive_errors += 1
        self.total_errors += 1
        self.total_requests += 1
        self.last_error_time = time.monotonic()

        if self.consecutive_errors >= 5:
            self.health = ProviderHealth.UNHEALTHY
            logger.warning(f"Provider {self.name} marked unhealthy after {self.consecutive_errors} errors")
        elif self.consecutive_errors >= 2:
            self.health = ProviderHealth.DEGRADED

    @property
    def error_rate(self) -> float:
        """Error rate as percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.total_errors / self.total_requests) * 100

    @property
    def min_request_interval(self) -> float:
        """Minimum seconds between requests"""
        return 1.0 / self.rate_limit


@dataclass
class OrderFilledEvent:
    """Parsed OrderFilled event from blockchain"""
    block_number: int
    transaction_hash: str
    log_index: int
    contract_address: str
    order_hash: str
    maker: str  # wallet address
    taker: str  # wallet address
    maker_asset_id: int
    taker_asset_id: int
    maker_amount_filled: int
    taker_amount_filled: int
    fee: int
    timestamp: Optional[int] = None  # block timestamp if available

    @property
    def wallets(self) -> Set[str]:
        """Return both maker and taker addresses"""
        return {self.maker.lower(), self.taker.lower()}


# Default free RPC providers for Polygon (no API key required)
# Note: Free tier limits vary and may change. Check provider docs for current limits.
DEFAULT_PROVIDERS = [
    RPCProvider(
        name="1RPC",
        url="https://1rpc.io/matic",
        rate_limit=10.0,
        max_block_range=5000,
        priority=5,
    ),
    RPCProvider(
        name="Polygon",
        url="https://polygon-rpc.com",
        rate_limit=5.0,
        max_block_range=1000,
        priority=4,
    ),
    RPCProvider(
        name="PublicNode",
        url="https://polygon-bor-rpc.publicnode.com",
        rate_limit=10.0,
        max_block_range=5000,
        priority=3,
    ),
    RPCProvider(
        name="Chainstack",
        url="https://polygon-mainnet.g.alchemy.com/v2/demo",
        rate_limit=5.0,
        max_block_range=2000,
        priority=2,
    ),
    RPCProvider(
        name="DrpcPublic",
        url="https://polygon.drpc.org",
        rate_limit=10.0,
        max_block_range=5000,
        priority=1,
    ),
]


class ProviderPool:
    """
    Manages a pool of RPC providers with rotation, health checks, and failover.

    Features:
    - Round-robin rotation across healthy providers
    - Automatic health tracking based on errors
    - Per-provider rate limiting
    - Fallback to degraded providers when all healthy are busy
    """

    def __init__(self, providers: List[RPCProvider] = None):
        self.providers = providers or [RPCProvider(**p.__dict__) for p in DEFAULT_PROVIDERS]
        self._current_index = 0
        self._lock = asyncio.Lock()

    @property
    def healthy_providers(self) -> List[RPCProvider]:
        """Get all healthy providers sorted by priority"""
        return sorted(
            [p for p in self.providers if p.health == ProviderHealth.HEALTHY],
            key=lambda p: p.priority,
            reverse=True,
        )

    @property
    def available_providers(self) -> List[RPCProvider]:
        """Get all non-unhealthy providers"""
        return sorted(
            [p for p in self.providers if p.health != ProviderHealth.UNHEALTHY],
            key=lambda p: (p.health == ProviderHealth.HEALTHY, p.priority),
            reverse=True,
        )

    @property
    def combined_rate_limit(self) -> float:
        """Total combined rate limit across all healthy providers"""
        return sum(p.rate_limit for p in self.healthy_providers)

    async def get_provider(self) -> Optional[RPCProvider]:
        """
        Get the next available provider, respecting rate limits.
        Returns None if no provider is available.
        """
        async with self._lock:
            available = self.available_providers
            if not available:
                # Try to recover unhealthy providers after cooldown
                now = time.monotonic()
                for p in self.providers:
                    if p.health == ProviderHealth.UNHEALTHY and now - p.last_error_time > 60:
                        p.health = ProviderHealth.DEGRADED
                        logger.info(f"Provider {p.name} cooldown expired, trying again")
                available = self.available_providers

            if not available:
                return None

            # Find provider that's ready (rate limit respected)
            now = time.monotonic()
            for provider in available:
                elapsed = now - provider.last_request_time
                if elapsed >= provider.min_request_interval:
                    provider.last_request_time = now
                    return provider

            # All providers are rate-limited, return the one with shortest wait
            best = min(available, key=lambda p: p.last_request_time + p.min_request_interval)
            wait_time = (best.last_request_time + best.min_request_interval) - now
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            best.last_request_time = time.monotonic()
            return best

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all providers"""
        return {
            "total_providers": len(self.providers),
            "healthy": len(self.healthy_providers),
            "combined_rate_limit": self.combined_rate_limit,
            "providers": [
                {
                    "name": p.name,
                    "health": p.health.value,
                    "requests": p.total_requests,
                    "errors": p.total_errors,
                    "error_rate": f"{p.error_rate:.1f}%",
                }
                for p in self.providers
            ],
        }


class PolygonRPCClient:
    """
    Low-level client for Polygon RPC queries.

    Handles JSON-RPC protocol, log parsing, and block timestamp lookups.
    """

    def __init__(
        self,
        provider_pool: ProviderPool = None,
        timeout: int = 30,
    ):
        self.pool = provider_pool or ProviderPool()
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._request_id = 0
        self._session: Optional[aiohttp.ClientSession] = None
        self._block_timestamps: Dict[int, int] = {}  # Cache block -> timestamp

    async def __aenter__(self):
        # Avoid brotli encoding issues with some RPC providers
        headers = {
            "Accept-Encoding": "gzip, deflate",
            "Content-Type": "application/json",
        }
        self._session = aiohttp.ClientSession(timeout=self.timeout, headers=headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()
            self._session = None

    def _next_request_id(self) -> int:
        """Get next JSON-RPC request ID"""
        self._request_id += 1
        return self._request_id

    async def _rpc_call(
        self,
        method: str,
        params: List[Any],
        provider: RPCProvider = None,
    ) -> Optional[Any]:
        """
        Make a JSON-RPC call to a provider.

        Returns the result field from the response, or None on error.
        """
        if provider is None:
            provider = await self.pool.get_provider()
            if provider is None:
                logger.error("No available RPC providers")
                return None

        payload = {
            "jsonrpc": "2.0",
            "id": self._next_request_id(),
            "method": method,
            "params": params,
        }

        try:
            async with self._session.post(
                provider.url,
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status != 200:
                    logger.warning(f"Provider {provider.name} returned HTTP {response.status}")
                    provider.mark_error()
                    return None

                data = await response.json()

                if "error" in data:
                    error = data["error"]
                    logger.warning(f"RPC error from {provider.name}: {error.get('message', error)}")
                    provider.mark_error()
                    return None

                provider.mark_success()
                return data.get("result")

        except asyncio.TimeoutError:
            logger.warning(f"Timeout from provider {provider.name}")
            provider.mark_error()
            return None
        except Exception as e:
            logger.warning(f"Error from provider {provider.name}: {e}")
            provider.mark_error()
            return None

    async def get_latest_block(self) -> Optional[int]:
        """Get the latest block number, trying multiple providers if needed"""
        # Try each available provider until one succeeds
        for provider in self.pool.available_providers:
            result = await self._rpc_call("eth_blockNumber", [], provider)
            if result:
                return int(result, 16)
        return None

    async def get_block_timestamp(self, block_number: int) -> Optional[int]:
        """Get timestamp for a block (cached)"""
        if block_number in self._block_timestamps:
            return self._block_timestamps[block_number]

        result = await self._rpc_call(
            "eth_getBlockByNumber",
            [hex(block_number), False],
        )
        if result and "timestamp" in result:
            timestamp = int(result["timestamp"], 16)
            self._block_timestamps[block_number] = timestamp
            return timestamp
        return None

    async def query_logs(
        self,
        from_block: int,
        to_block: int,
        contract_address: str,
        topics: List[str] = None,
        provider: RPCProvider = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Query event logs from a contract.

        Args:
            from_block: Start block (inclusive)
            to_block: End block (inclusive)
            contract_address: Contract to query
            topics: Event topic filters
            provider: Specific provider to use (optional)

        Returns:
            List of raw log entries, or None on error
        """
        params = {
            "fromBlock": hex(from_block),
            "toBlock": hex(to_block),
            "address": contract_address,
        }
        if topics:
            params["topics"] = topics

        return await self._rpc_call("eth_getLogs", [params], provider)

    async def query_order_filled_events(
        self,
        from_block: int,
        to_block: int,
        contract_address: str = CTF_EXCHANGE_MULTI,
    ) -> List[OrderFilledEvent]:
        """
        Query OrderFilled events from CTF Exchange contract.

        Args:
            from_block: Start block
            to_block: End block
            contract_address: CTF Exchange contract address

        Returns:
            List of parsed OrderFilledEvent objects
        """
        logs = await self.query_logs(
            from_block=from_block,
            to_block=to_block,
            contract_address=contract_address,
            topics=[ORDER_FILLED_TOPIC],
        )

        if logs is None:
            return []

        events = []
        for log in logs:
            try:
                event = self.parse_order_filled(log)
                if event:
                    events.append(event)
            except Exception as e:
                logger.debug(f"Error parsing log: {e}")

        return events

    @staticmethod
    def parse_order_filled(raw_log: Dict[str, Any]) -> Optional[OrderFilledEvent]:
        """
        Parse a raw log entry into an OrderFilledEvent.

        The OrderFilled event has 3 indexed parameters (in topics) and 5 non-indexed (in data).

        Topics:
            [0] = event signature
            [1] = orderHash (bytes32, indexed)
            [2] = maker (address, indexed)
            [3] = taker (address, indexed)

        Data (packed, 5 * 32 bytes = 160 bytes):
            [0:32]   = makerAssetId (uint256) - but often 0 and asset stored differently
            [32:64]  = takerAssetId (bytes32) - condition token ID
            [64:96]  = makerAmountFilled (uint256)
            [96:128] = takerAmountFilled (uint256)
            [128:160] = fee (uint256)
        """
        try:
            topics = raw_log.get("topics", [])
            data = raw_log.get("data", "0x")

            if len(topics) < 4:
                return None

            # Extract indexed parameters from topics
            order_hash = topics[1]
            # Address is in last 40 chars of 64-char topic (padded to 32 bytes)
            maker = "0x" + topics[2][-40:]
            taker = "0x" + topics[3][-40:]

            # Parse non-indexed data
            # Data should have at least 5 uint256 values = 320 hex chars (after 0x)
            data_hex = data[2:] if data.startswith("0x") else data

            # Handle variable data lengths gracefully
            # Minimum: need at least makerAssetId and takerAssetId (128 hex chars)
            if len(data_hex) < 128:
                # Minimal parse - just get addresses
                return OrderFilledEvent(
                    block_number=int(raw_log.get("blockNumber", "0x0"), 16),
                    transaction_hash=raw_log.get("transactionHash", ""),
                    log_index=int(raw_log.get("logIndex", "0x0"), 16),
                    contract_address=raw_log.get("address", ""),
                    order_hash=order_hash,
                    maker=maker,
                    taker=taker,
                    maker_asset_id=0,
                    taker_asset_id=0,
                    maker_amount_filled=0,
                    taker_amount_filled=0,
                    fee=0,
                )

            # Parse 32-byte chunks (64 hex chars each)
            def parse_uint256(offset: int) -> int:
                start = offset * 64
                end = start + 64
                if end <= len(data_hex):
                    return int(data_hex[start:end], 16)
                return 0

            maker_asset_id = parse_uint256(0)
            taker_asset_id = parse_uint256(1)
            maker_amount_filled = parse_uint256(2)
            taker_amount_filled = parse_uint256(3)
            fee = parse_uint256(4)

            return OrderFilledEvent(
                block_number=int(raw_log.get("blockNumber", "0x0"), 16),
                transaction_hash=raw_log.get("transactionHash", ""),
                log_index=int(raw_log.get("logIndex", "0x0"), 16),
                contract_address=raw_log.get("address", ""),
                order_hash=order_hash,
                maker=maker,
                taker=taker,
                maker_asset_id=maker_asset_id,
                taker_asset_id=taker_asset_id,
                maker_amount_filled=maker_amount_filled,
                taker_amount_filled=taker_amount_filled,
                fee=fee,
            )
        except Exception as e:
            logger.debug(f"Failed to parse OrderFilled event: {e}")
            return None

    async def extract_wallets_from_range(
        self,
        from_block: int,
        to_block: int,
        contracts: List[str] = None,
    ) -> Set[str]:
        """
        Extract all unique wallet addresses from OrderFilled events in a block range.

        Args:
            from_block: Start block
            to_block: End block
            contracts: List of contract addresses to query (defaults to both CTF exchanges)

        Returns:
            Set of unique wallet addresses (lowercased)
        """
        if contracts is None:
            contracts = [CTF_EXCHANGE_MULTI, CTF_EXCHANGE_LEGACY]

        wallets = set()

        for contract in contracts:
            events = await self.query_order_filled_events(from_block, to_block, contract)
            for event in events:
                wallets.update(event.wallets)

        return wallets
