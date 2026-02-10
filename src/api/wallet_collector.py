"""
Comprehensive wallet collector - aggregates wallets from all Goldsky subgraph sources.

Combines:
- Orderbook (trades): maker and taker addresses
- PnL (positions): user addresses
- Activity (redemptions, splits, merges): participant addresses

This bypasses the single-source pagination limit by querying multiple subgraphs.
"""
import asyncio
import aiohttp
import time
from typing import Set, Dict, List, Tuple
from dataclasses import dataclass


GOLDSKY_PROJECT_ID = "project_cl6mb8i9h0003e201j6li0diw"
GOLDSKY_BASE_URL = f"https://api.goldsky.com/api/public/{GOLDSKY_PROJECT_ID}/subgraphs"

ENDPOINTS = {
    "activity": f"{GOLDSKY_BASE_URL}/activity-subgraph/0.0.4/gn",
    "pnl": f"{GOLDSKY_BASE_URL}/pnl-subgraph/0.0.14/gn",
    "orderbook": f"{GOLDSKY_BASE_URL}/orderbook-subgraph/0.0.1/gn",
}


@dataclass
class CollectionStats:
    """Stats from wallet collection."""
    total_wallets: int
    orderbook_wallets: int
    pnl_wallets: int
    activity_wallets: int
    duration_seconds: float
    requests_made: int


async def _query(
    session: aiohttp.ClientSession,
    endpoint: str,
    query: str,
    variables: dict,
    retries: int = 3,
) -> dict:
    """Execute GraphQL query with retries."""
    for attempt in range(retries):
        try:
            async with session.post(
                endpoint,
                json={"query": query, "variables": variables},
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if "errors" not in data:
                        return data.get("data", {})
                elif resp.status == 429:
                    await asyncio.sleep(2 ** attempt)
                    continue
        except asyncio.TimeoutError:
            if attempt < retries - 1:
                await asyncio.sleep(1)
        except Exception:
            if attempt < retries - 1:
                await asyncio.sleep(0.5)
    return {}


async def _collect_from_orderbook(
    session: aiohttp.ClientSession,
    max_skip: int = 500000,
) -> Tuple[Set[str], int]:
    """Collect wallets from orderbook (trades)."""
    q = """query($first: Int!, $skip: Int!) {
        orderFilledEvents(first: $first, skip: $skip, orderBy: timestamp, orderDirection: desc) {
            maker
            taker
        }
    }"""

    wallets = set()
    requests = 0
    skip = 0

    while skip < max_skip:
        data = await _query(session, ENDPOINTS["orderbook"], q, {"first": 1000, "skip": skip})
        events = data.get("orderFilledEvents", [])
        requests += 1

        if not events:
            break

        for e in events:
            m = e.get("maker", "").lower()
            t = e.get("taker", "").lower()
            if m and len(m) == 42:
                wallets.add(m)
            if t and len(t) == 42:
                wallets.add(t)

        skip += 1000

        # Rate limiting
        if requests % 10 == 0:
            await asyncio.sleep(0.1)

    return wallets, requests


async def _collect_from_pnl(
    session: aiohttp.ClientSession,
    max_skip: int = 500000,
) -> Tuple[Set[str], int]:
    """Collect wallets from PnL subgraph (positions)."""
    q = """query($first: Int!, $skip: Int!) {
        userPositions(first: $first, skip: $skip) {
            user
        }
    }"""

    wallets = set()
    requests = 0
    skip = 0

    while skip < max_skip:
        data = await _query(session, ENDPOINTS["pnl"], q, {"first": 1000, "skip": skip})
        positions = data.get("userPositions", [])
        requests += 1

        if not positions:
            break

        for p in positions:
            w = p.get("user", "").lower()
            if w and len(w) == 42:
                wallets.add(w)

        skip += 1000

        if requests % 10 == 0:
            await asyncio.sleep(0.1)

    return wallets, requests


async def _collect_from_activity(
    session: aiohttp.ClientSession,
    max_skip: int = 500000,
) -> Tuple[Set[str], int]:
    """Collect wallets from activity subgraph (redemptions, splits, merges)."""
    queries = [
        # Redemptions
        ("""query($first: Int!, $skip: Int!) {
            redemptions(first: $first, skip: $skip) { redeemer }
        }""", "redemptions", "redeemer"),

        # Splits
        ("""query($first: Int!, $skip: Int!) {
            splits(first: $first, skip: $skip) { stakeholder }
        }""", "splits", "stakeholder"),

        # Merges
        ("""query($first: Int!, $skip: Int!) {
            merges(first: $first, skip: $skip) { stakeholder }
        }""", "merges", "stakeholder"),
    ]

    wallets = set()
    total_requests = 0

    for q, entity_name, field in queries:
        skip = 0
        while skip < max_skip:
            data = await _query(session, ENDPOINTS["activity"], q, {"first": 1000, "skip": skip})
            items = data.get(entity_name, [])
            total_requests += 1

            if not items:
                break

            for item in items:
                w = item.get(field, "").lower()
                if w and len(w) == 42:
                    wallets.add(w)

            skip += 1000

            if total_requests % 10 == 0:
                await asyncio.sleep(0.1)

    return wallets, total_requests


async def collect_all_wallets(
    max_skip_per_source: int = 500000,
    verbose: bool = True,
) -> Tuple[Set[str], CollectionStats]:
    """
    Collect wallets from all available Goldsky subgraph sources.

    Args:
        max_skip_per_source: Maximum pagination depth per source
        verbose: Print progress

    Returns:
        Tuple of (wallet set, collection stats)
    """
    start = time.time()

    connector = aiohttp.TCPConnector(limit=20, force_close=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Run all collectors in parallel
        if verbose:
            print("Collecting wallets from all sources...")

        tasks = [
            _collect_from_orderbook(session, max_skip_per_source),
            _collect_from_pnl(session, max_skip_per_source),
            _collect_from_activity(session, max_skip_per_source),
        ]

        results = await asyncio.gather(*tasks)

        orderbook_wallets, orderbook_requests = results[0]
        pnl_wallets, pnl_requests = results[1]
        activity_wallets, activity_requests = results[2]

        # Combine all wallets
        all_wallets = orderbook_wallets | pnl_wallets | activity_wallets

        elapsed = time.time() - start
        total_requests = orderbook_requests + pnl_requests + activity_requests

        stats = CollectionStats(
            total_wallets=len(all_wallets),
            orderbook_wallets=len(orderbook_wallets),
            pnl_wallets=len(pnl_wallets),
            activity_wallets=len(activity_wallets),
            duration_seconds=elapsed,
            requests_made=total_requests,
        )

        if verbose:
            print(f"\nCollection complete:")
            print(f"  Orderbook: {stats.orderbook_wallets:,} wallets")
            print(f"  PnL: {stats.pnl_wallets:,} wallets")
            print(f"  Activity: {stats.activity_wallets:,} wallets")
            print(f"  TOTAL: {stats.total_wallets:,} unique wallets")
            print(f"  Time: {elapsed:.1f}s ({total_requests} requests)")

        return all_wallets, stats


async def collect_wallets_with_trades(
    max_skip: int = 500000,
    verbose: bool = True,
) -> Tuple[Set[str], Dict[str, List[dict]]]:
    """
    Collect wallets AND their trade data from orderbook.

    This is more efficient when you need both the wallet list and trade data.

    Returns:
        Tuple of (wallet set, trades dict mapping address -> list of trade dicts)
    """
    q = """query($first: Int!, $skip: Int!) {
        orderFilledEvents(first: $first, skip: $skip, orderBy: timestamp, orderDirection: desc) {
            id
            timestamp
            maker
            taker
            makerAssetId
            takerAssetId
            makerAmountFilled
            takerAmountFilled
            fee
        }
    }"""

    wallets = set()
    trades_by_wallet: Dict[str, List[dict]] = {}

    connector = aiohttp.TCPConnector(limit=20, force_close=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        skip = 0
        requests = 0

        while skip < max_skip:
            data = await _query(session, ENDPOINTS["orderbook"], q, {"first": 1000, "skip": skip})
            events = data.get("orderFilledEvents", [])
            requests += 1

            if not events:
                break

            for e in events:
                maker = e.get("maker", "").lower()
                taker = e.get("taker", "").lower()

                if maker and len(maker) == 42:
                    wallets.add(maker)
                    if maker not in trades_by_wallet:
                        trades_by_wallet[maker] = []
                    trades_by_wallet[maker].append({**e, "_role": "maker"})

                if taker and len(taker) == 42:
                    wallets.add(taker)
                    if taker not in trades_by_wallet:
                        trades_by_wallet[taker] = []
                    trades_by_wallet[taker].append({**e, "_role": "taker"})

            skip += 1000

            if verbose and requests % 100 == 0:
                print(f"  Fetched {skip:,} events, {len(wallets):,} wallets...")

            if requests % 10 == 0:
                await asyncio.sleep(0.1)

    if verbose:
        total_trades = sum(len(t) for t in trades_by_wallet.values())
        print(f"\nCollected {len(wallets):,} wallets with {total_trades:,} trades")

    return wallets, trades_by_wallet


if __name__ == "__main__":
    # Test collection
    async def main():
        print("=" * 60)
        print("COMPREHENSIVE WALLET COLLECTION")
        print("=" * 60)

        # Quick test with limited pagination
        wallets, stats = await collect_all_wallets(
            max_skip_per_source=100000,  # 100k max per source for quick test
            verbose=True,
        )

        print(f"\n{'=' * 60}")
        print(f"READY TO PROCESS: {len(wallets):,} wallets")

    asyncio.run(main())
