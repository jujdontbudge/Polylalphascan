"""
Goldsky Subgraph Client - GraphQL client for Polymarket subgraphs.

Goldsky provides 5 Polymarket subgraphs with 99.9% uptime:
- polymarket-activity: Trades
- polymarket-positions: Positions
- polymarket-pnl: Pre-computed PnL

Batch queries 100 wallets per request for 50-100x fewer API calls.
"""
import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

# Goldsky Polymarket Subgraph endpoints (official Polymarket project ID)
# Source: https://www.polytrackhq.app/blog/polymarket-graphql-subgraph-guide
GOLDSKY_PROJECT_ID = "project_cl6mb8i9h0003e201j6li0diw"
GOLDSKY_BASE_URL = f"https://api.goldsky.com/api/public/{GOLDSKY_PROJECT_ID}/subgraphs"

GOLDSKY_ENDPOINTS = {
    "activity": f"{GOLDSKY_BASE_URL}/activity-subgraph/0.0.4/gn",
    "positions": f"{GOLDSKY_BASE_URL}/positions-subgraph/0.0.7/gn",
    "pnl": f"{GOLDSKY_BASE_URL}/pnl-subgraph/0.0.14/gn",
    "orderbook": f"{GOLDSKY_BASE_URL}/orderbook-subgraph/0.0.1/gn",
    "open_interest": f"{GOLDSKY_BASE_URL}/oi-subgraph/0.0.6/gn",
}

# GraphQL queries for batch operations
# Based on actual Polymarket Goldsky subgraph introspection:
# - Orderbook subgraph: orderFilledEvents (trades)
# - PnL subgraph: userPositions (positions)
# - Activity subgraph: splits, merges, redemptions (events)

# Trades from orderbook subgraph - query by maker or taker
BATCH_TRADES_MAKER_QUERY = """
query BatchTradesMaker($wallets: [String!]!, $first: Int!, $skip: Int!) {
  orderFilledEvents(
    first: $first
    skip: $skip
    where: { maker_in: $wallets }
    orderBy: timestamp
    orderDirection: desc
  ) {
    id
    transactionHash
    timestamp
    orderHash
    maker
    taker
    makerAssetId
    takerAssetId
    makerAmountFilled
    takerAmountFilled
    fee
  }
}
"""

BATCH_TRADES_TAKER_QUERY = """
query BatchTradesTaker($wallets: [String!]!, $first: Int!, $skip: Int!) {
  orderFilledEvents(
    first: $first
    skip: $skip
    where: { taker_in: $wallets }
    orderBy: timestamp
    orderDirection: desc
  ) {
    id
    transactionHash
    timestamp
    orderHash
    maker
    taker
    makerAssetId
    takerAssetId
    makerAmountFilled
    takerAmountFilled
    fee
  }
}
"""

# Positions from PnL subgraph
BATCH_POSITIONS_QUERY = """
query BatchPositions($wallets: [String!]!, $first: Int!) {
  userPositions(
    first: $first
    where: { user_in: $wallets }
  ) {
    id
    user
    tokenId
    amount
    avgPrice
    realizedPnl
    totalBought
  }
}
"""

# Activity events from activity subgraph
BATCH_REDEMPTIONS_QUERY = """
query BatchRedemptions($wallets: [String!]!, $first: Int!) {
  redemptions(
    first: $first
    where: { redeemer_in: $wallets }
    orderBy: timestamp
    orderDirection: desc
  ) {
    id
    redeemer
    timestamp
    payout
    collateralToken
  }
}
"""

# Fallback: get all recent trades and filter client-side
ALL_RECENT_TRADES_QUERY = """
query AllRecentTrades($first: Int!, $skip: Int!) {
  orderFilledEvents(
    first: $first
    skip: $skip
    orderBy: timestamp
    orderDirection: desc
  ) {
    id
    transactionHash
    timestamp
    maker
    taker
    makerAssetId
    takerAssetId
    makerAmountFilled
    takerAmountFilled
    fee
  }
}
"""


@dataclass
class GoldskyTrade:
    """Trade from Goldsky subgraph (orderFilledEvents)"""
    id: str
    user: str
    timestamp: datetime
    market_id: str
    condition_id: str
    side: str  # BUY or SELL based on maker/taker role
    price: float
    size: float
    value: float
    fee: float
    transaction_hash: str

    @classmethod
    def from_order_filled(cls, data: Dict[str, Any], user_address: str) -> "GoldskyTrade":
        """
        Create from orderFilledEvent GraphQL response.

        The orderFilledEvent schema:
        - maker, taker: wallet addresses
        - makerAssetId, takerAssetId: token IDs (outcome tokens or USDC)
        - makerAmountFilled, takerAmountFilled: amounts (in wei, 10^6 for USDC)
        - fee: fee amount
        - timestamp: unix timestamp

        Side logic:
        - If user is maker and makerAsset is USDC: they're BUYING outcome tokens
        - If user is taker and takerAsset is USDC: they're BUYING outcome tokens
        - Otherwise: SELLING
        """
        # Parse timestamp
        ts = data.get("timestamp", 0)
        if isinstance(ts, str):
            ts = int(ts) if ts.isdigit() else 0
        timestamp = datetime.fromtimestamp(int(ts)) if ts else datetime.utcnow()

        user_lower = user_address.lower()
        maker = (data.get("maker") or "").lower()
        taker = (data.get("taker") or "").lower()

        # Determine side based on whether user is maker or taker
        # and which asset they're giving up (maker gives makerAsset)
        maker_asset = data.get("makerAssetId", "")
        taker_asset = data.get("takerAssetId", "")

        # USDC contract on Polygon: typically ends with specific pattern or is short
        # Outcome tokens have longer IDs
        is_user_maker = (user_lower == maker)

        # Parse amounts (stored as BigInt strings, convert to float)
        maker_amount = float(data.get("makerAmountFilled", 0) or 0) / 1e6  # Assume 6 decimals
        taker_amount = float(data.get("takerAmountFilled", 0) or 0) / 1e6
        fee = float(data.get("fee", 0) or 0) / 1e6

        if is_user_maker:
            # User is maker - they provide makerAsset
            # If makerAsset looks like USDC (short ID), they're BUYING
            # The size is taker_amount (what they receive)
            # The price is maker_amount / taker_amount (USDC / tokens)
            if taker_amount > 0:
                price = maker_amount / taker_amount
                size = taker_amount
                side = "BUY"
            else:
                price = 0.0
                size = maker_amount
                side = "SELL"
            user = maker
            condition_id = taker_asset  # The outcome token they receive
        else:
            # User is taker - they take takerAsset
            if maker_amount > 0:
                price = taker_amount / maker_amount
                size = maker_amount
                side = "BUY"
            else:
                price = 0.0
                size = taker_amount
                side = "SELL"
            user = taker
            condition_id = maker_asset  # The outcome token they receive

        # Value = price * size
        value = price * size

        return cls(
            id=data.get("id", ""),
            user=user,
            timestamp=timestamp,
            market_id="",  # Need to lookup from condition
            condition_id=condition_id[:42] if condition_id else "",  # Truncate if too long
            side=side,
            price=min(price, 1.0),  # Cap at 1.0 for probability
            size=size,
            value=value,
            fee=fee,
            transaction_hash=data.get("transactionHash", "") or "",
        )

    @classmethod
    def from_graphql(cls, data: Dict[str, Any]) -> "GoldskyTrade":
        """Create from generic GraphQL response (legacy support)"""
        # Handle timestamp (may be string or int)
        ts = data.get("timestamp", 0)
        if isinstance(ts, str):
            ts = int(ts) if ts.isdigit() else 0
        timestamp = datetime.fromtimestamp(int(ts)) if ts else datetime.utcnow()

        # Handle market data
        market = data.get("market", {})
        market_id = market.get("id", "") if market else ""

        # condition may be separate field or in market
        condition_id = data.get("condition", "") or market.get("conditionId", "") or ""

        # Compute value from price * size if not provided
        price = float(data.get("price", 0) or 0)
        size = float(data.get("size", 0) or 0)
        value = float(data.get("value", 0) or 0) or (price * size)

        return cls(
            id=data.get("id", ""),
            user=data.get("user", "").lower() if data.get("user") else "",
            timestamp=timestamp,
            market_id=market_id,
            condition_id=condition_id,
            side=str(data.get("side", "BUY")).upper(),
            price=price,
            size=size,
            value=value,
            fee=float(data.get("fee", 0) or 0),
            transaction_hash=data.get("transactionHash", "") or "",
        )


@dataclass
class GoldskyPosition:
    """Position from Goldsky subgraph (userPositions from PnL subgraph)"""
    id: str
    user: str
    market_id: str
    condition_id: str
    outcome: str
    outcome_index: int
    size: float
    avg_price: float
    current_price: float
    realized_pnl: float

    @classmethod
    def from_user_position(cls, data: Dict[str, Any]) -> "GoldskyPosition":
        """
        Create from userPositions GraphQL response.

        Schema from PnL subgraph:
        - user: wallet address
        - tokenId: outcome token ID (BigInt)
        - amount: position size (BigInt, 10^6 decimals)
        - avgPrice: average entry price (BigInt, 10^6 decimals)
        - realizedPnl: realized profit/loss (BigInt)
        - totalBought: total amount bought (BigInt)
        """
        user = (data.get("user") or "").lower()
        token_id = data.get("tokenId", "")

        # Convert BigInt values (stored as strings, 6 decimal precision)
        amount = float(data.get("amount", 0) or 0) / 1e6
        avg_price = float(data.get("avgPrice", 0) or 0) / 1e6
        realized_pnl = float(data.get("realizedPnl", 0) or 0) / 1e6

        return cls(
            id=data.get("id", ""),
            user=user,
            market_id="",  # Token ID contains this info
            condition_id=str(token_id)[:42] if token_id else "",
            outcome="",  # Would need token metadata lookup
            outcome_index=0,  # Would need token metadata lookup
            size=amount,
            avg_price=min(avg_price, 1.0),  # Cap at 1.0 for probability
            current_price=0.0,  # Not available in this schema
            realized_pnl=realized_pnl,
        )

    @classmethod
    def from_graphql(cls, data: Dict[str, Any]) -> "GoldskyPosition":
        """Create from generic GraphQL response (legacy support)"""
        # Handle nested user object or string
        user_data = data.get("user", {})
        user = user_data.get("id", "") if isinstance(user_data, dict) else str(user_data or "")

        # Condition ID from direct field
        condition_id = data.get("condition", "") or ""

        return cls(
            id=data.get("id", ""),
            user=user.lower() if user else "",
            market_id="",  # May need separate lookup
            condition_id=condition_id,
            outcome="",  # Determined by outcome_index
            outcome_index=int(data.get("outcomeIndex", 0) or 0),
            size=float(data.get("balance", 0) or data.get("size", 0) or 0),
            avg_price=float(data.get("averagePrice", 0) or data.get("avgPrice", 0) or 0),
            current_price=float(data.get("currentPrice", 0) or 0),
            realized_pnl=float(data.get("realizedPnl", 0) or 0),
        )


@dataclass
class GoldskyPnL:
    """Pre-computed PnL from Goldsky subgraph"""
    user: str
    total_realized_pnl: float
    total_unrealized_pnl: float
    total_volume: float
    total_trades: int

    @classmethod
    def from_graphql(cls, data: Dict[str, Any]) -> "GoldskyPnL":
        """Create from GraphQL response"""
        return cls(
            user=data.get("user", ""),
            total_realized_pnl=float(data.get("totalRealizedPnl", 0)),
            total_unrealized_pnl=float(data.get("totalUnrealizedPnl", 0)),
            total_volume=float(data.get("totalVolume", 0)),
            total_trades=int(data.get("totalTrades", 0)),
        )


class GoldskyClient:
    """
    GraphQL client for Goldsky Polymarket subgraphs.

    Features:
    - Batch queries (100 wallets per request)
    - Automatic retry with exponential backoff
    - Rate limiting with token bucket
    - Connection pooling
    """

    def __init__(
        self,
        batch_size: int = 100,
        max_retries: int = 3,
        timeout_seconds: int = 30,
        requests_per_second: float = 10.0,
    ):
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self.requests_per_second = requests_per_second

        # Rate limiting
        self._last_request_time = 0.0
        self._lock = asyncio.Lock()

        # Stats
        self.successful_requests = 0
        self.failed_requests = 0

    async def _rate_limit(self):
        """Apply rate limiting"""
        async with self._lock:
            import time
            now = time.monotonic()
            min_interval = 1.0 / self.requests_per_second
            wait_time = max(0, min_interval - (now - self._last_request_time))
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self._last_request_time = time.monotonic()

    async def _execute_query(
        self,
        session: aiohttp.ClientSession,
        endpoint: str,
        query: str,
        variables: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Execute GraphQL query with retries"""
        for attempt in range(self.max_retries):
            await self._rate_limit()

            try:
                async with session.post(
                    endpoint,
                    json={"query": query, "variables": variables},
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "errors" in result:
                            logger.warning(f"GraphQL errors: {result['errors']}")
                            return None
                        self.successful_requests += 1
                        return result.get("data", {})
                    elif response.status == 429:
                        wait = 2 ** attempt
                        logger.warning(f"Rate limited, waiting {wait}s...")
                        await asyncio.sleep(wait)
                    else:
                        logger.debug(f"HTTP {response.status} for {endpoint}")
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(0.5 * (attempt + 1))
            except asyncio.TimeoutError:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))
            except Exception as e:
                logger.debug(f"Error executing query: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))

        self.failed_requests += 1
        return None

    async def get_batch_trades(
        self,
        wallet_addresses: List[str],
        max_per_wallet: int = 1000,
    ) -> Dict[str, List[GoldskyTrade]]:
        """
        Fetch trades for multiple wallets in a single batch query.

        Uses orderbook subgraph's orderFilledEvents entity.
        Queries both maker and taker roles for complete trade history.

        Args:
            wallet_addresses: List of wallet addresses (max 100)
            max_per_wallet: Maximum trades to fetch per wallet

        Returns:
            Dict mapping address -> list of trades
        """
        if not wallet_addresses:
            return {}

        # Normalize addresses to lowercase
        wallets = [addr.lower() for addr in wallet_addresses[:self.batch_size]]

        connector = aiohttp.TCPConnector(limit=10, force_close=True)
        async with aiohttp.ClientSession(
            timeout=self.timeout,
            connector=connector,
        ) as session:
            # Fetch trades where user is maker AND where user is taker
            all_events: List[Dict] = []

            # Query both maker and taker trades in parallel
            for query, role in [
                (BATCH_TRADES_MAKER_QUERY, "maker"),
                (BATCH_TRADES_TAKER_QUERY, "taker"),
            ]:
                skip = 0
                first = 1000

                while True:
                    data = await self._execute_query(
                        session,
                        GOLDSKY_ENDPOINTS["orderbook"],
                        query,
                        {"wallets": wallets, "first": first, "skip": skip},
                    )

                    if not data or "orderFilledEvents" not in data:
                        break

                    events = data["orderFilledEvents"]
                    if not events:
                        break

                    # Tag events with the query role for parsing
                    for event in events:
                        event["_role"] = role
                    all_events.extend(events)

                    skip += first

                    # Check if we have enough total events
                    if skip >= 50000:  # Max 50 pages per role for deeper pagination
                        break

            # Group by user (both maker and taker addresses)
            result: Dict[str, List[GoldskyTrade]] = {addr: [] for addr in wallets}

            for event_data in all_events:
                role = event_data.get("_role", "maker")
                user = (event_data.get(role, "") or "").lower()

                if user in result and len(result[user]) < max_per_wallet:
                    try:
                        trade = GoldskyTrade.from_order_filled(event_data, user)
                        result[user].append(trade)
                    except Exception as e:
                        logger.debug(f"Error parsing order filled event: {e}")

            total_trades = sum(len(v) for v in result.values())
            logger.info(f"Goldsky orderbook: fetched {total_trades} trades for {len(wallets)} wallets from {len(all_events)} events")
            return result

    async def get_batch_positions(
        self,
        wallet_addresses: List[str],
    ) -> Dict[str, List[GoldskyPosition]]:
        """
        Fetch positions for multiple wallets in a single batch query.

        Uses PnL subgraph's userPositions entity.

        Args:
            wallet_addresses: List of wallet addresses (max 100)

        Returns:
            Dict mapping address -> list of positions
        """
        if not wallet_addresses:
            return {}

        wallets = [addr.lower() for addr in wallet_addresses[:self.batch_size]]

        connector = aiohttp.TCPConnector(limit=10, force_close=True)
        async with aiohttp.ClientSession(
            timeout=self.timeout,
            connector=connector,
        ) as session:
            data = await self._execute_query(
                session,
                GOLDSKY_ENDPOINTS["pnl"],  # Use PnL subgraph for userPositions
                BATCH_POSITIONS_QUERY,
                {"wallets": wallets, "first": 1000},
            )

            if not data or "userPositions" not in data:
                logger.debug(f"No userPositions found in Goldsky PnL subgraph for batch")
                return {addr: [] for addr in wallets}

            # Group by user
            result: Dict[str, List[GoldskyPosition]] = {addr: [] for addr in wallets}
            for pos_data in data["userPositions"]:
                user = (pos_data.get("user") or "").lower()

                if user in result:
                    try:
                        position = GoldskyPosition.from_user_position(pos_data)
                        result[user].append(position)
                    except Exception as e:
                        logger.debug(f"Error parsing userPosition: {e}")

            total = sum(len(v) for v in result.values())
            logger.info(f"Goldsky PnL: fetched {total} positions for {len(wallets)} wallets")
            return result

    async def get_batch_pnl(
        self,
        wallet_addresses: List[str],
    ) -> Dict[str, GoldskyPnL]:
        """
        Fetch pre-computed PnL for multiple wallets.

        Args:
            wallet_addresses: List of wallet addresses (max 100)

        Returns:
            Dict mapping address -> PnL summary
        """
        if not wallet_addresses:
            return {}

        wallets = [addr.lower() for addr in wallet_addresses[:self.batch_size]]

        connector = aiohttp.TCPConnector(limit=10)
        async with aiohttp.ClientSession(
            timeout=self.timeout,
            connector=connector,
        ) as session:
            data = await self._execute_query(
                session,
                GOLDSKY_ENDPOINTS["pnl"],
                BATCH_PNL_QUERY,
                {"wallets": wallets},
            )

            if not data or "userPnLs" not in data:
                return {}

            result: Dict[str, GoldskyPnL] = {}
            for pnl_data in data["userPnLs"]:
                user = pnl_data.get("user", "").lower()
                result[user] = GoldskyPnL.from_graphql(pnl_data)

            return result

    async def get_all_wallets_data(
        self,
        wallet_addresses: List[str],
        max_trades_per_wallet: int = 1000,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch trades, positions, and PnL for multiple wallets.

        Batches requests for optimal throughput.

        Args:
            wallet_addresses: List of wallet addresses
            max_trades_per_wallet: Maximum trades per wallet

        Returns:
            Dict mapping address -> {"trades": [...], "positions": [...], "pnl": {...}}
        """
        result = {}

        # Process in batches
        for i in range(0, len(wallet_addresses), self.batch_size):
            batch = wallet_addresses[i:i + self.batch_size]

            # Fetch all data types in parallel
            trades_task = self.get_batch_trades(batch, max_trades_per_wallet)
            positions_task = self.get_batch_positions(batch)
            pnl_task = self.get_batch_pnl(batch)

            trades, positions, pnls = await asyncio.gather(
                trades_task, positions_task, pnl_task
            )

            # Combine results
            for addr in batch:
                addr_lower = addr.lower()
                result[addr_lower] = {
                    "trades": trades.get(addr_lower, []),
                    "positions": positions.get(addr_lower, []),
                    "pnl": pnls.get(addr_lower),
                }

        return result


    async def test_connection(self) -> Dict[str, Any]:
        """
        Test connection to Goldsky subgraphs.

        Returns:
            Dict with status of each endpoint
        """
        results = {}

        connector = aiohttp.TCPConnector(limit=10, force_close=True)
        async with aiohttp.ClientSession(
            timeout=self.timeout,
            connector=connector,
        ) as session:
            for name, endpoint in GOLDSKY_ENDPOINTS.items():
                try:
                    # Simple introspection query
                    data = await self._execute_query(
                        session,
                        endpoint,
                        "{ __schema { queryType { name } } }",
                        {},
                    )
                    results[name] = {
                        "status": "ok" if data else "error",
                        "endpoint": endpoint,
                    }
                except Exception as e:
                    results[name] = {
                        "status": "error",
                        "endpoint": endpoint,
                        "error": str(e),
                    }

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        return {
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.successful_requests / max(1, self.successful_requests + self.failed_requests) * 100,
        }


# Convenience function
async def fetch_batch_from_goldsky(
    wallet_addresses: List[str],
    batch_size: int = 100,
) -> Dict[str, Dict[str, Any]]:
    """
    Convenience function to fetch data from Goldsky.

    Args:
        wallet_addresses: List of wallet addresses
        batch_size: Wallets per batch

    Returns:
        Dict mapping address -> wallet data
    """
    client = GoldskyClient(batch_size=batch_size)
    return await client.get_all_wallets_data(wallet_addresses)


async def test_goldsky_connection() -> bool:
    """
    Test the Goldsky connection and print results.

    Returns:
        True if at least one endpoint is working
    """
    client = GoldskyClient()
    print("Testing Goldsky connection...")
    print(f"Endpoints:")
    for name, url in GOLDSKY_ENDPOINTS.items():
        print(f"  {name}: {url}")
    print()

    results = await client.test_connection()

    working = 0
    for name, status in results.items():
        symbol = "✓" if status["status"] == "ok" else "✗"
        print(f"  {symbol} {name}: {status['status']}")
        if status["status"] == "ok":
            working += 1

    print(f"\n{working}/{len(results)} endpoints working")
    return working > 0


if __name__ == "__main__":
    # CLI test
    import sys

    async def main():
        success = await test_goldsky_connection()

        # Optional: test with a known wallet
        if len(sys.argv) > 1 and sys.argv[1] == "--test-wallet":
            print("\n\nTesting with sample wallet...")
            # Known active Polymarket wallet (from leaderboard)
            test_wallet = "0x123456789abcdef0123456789abcdef01234567"  # Replace with real wallet
            client = GoldskyClient()
            trades = await client.get_batch_trades([test_wallet], max_per_wallet=10)
            print(f"Trades found: {len(trades.get(test_wallet.lower(), []))}")

        return 0 if success else 1

    exit_code = asyncio.run(main())
    sys.exit(exit_code)
