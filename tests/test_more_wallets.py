"""
Test getting more wallets from multiple Goldsky subgraph sources.
"""
import asyncio
import aiohttp
import time
from collections import defaultdict

GOLDSKY_PROJECT_ID = "project_cl6mb8i9h0003e201j6li0diw"
GOLDSKY_BASE_URL = f"https://api.goldsky.com/api/public/{GOLDSKY_PROJECT_ID}/subgraphs"

ENDPOINTS = {
    "activity": f"{GOLDSKY_BASE_URL}/activity-subgraph/0.0.4/gn",
    "positions": f"{GOLDSKY_BASE_URL}/positions-subgraph/0.0.7/gn",
    "pnl": f"{GOLDSKY_BASE_URL}/pnl-subgraph/0.0.14/gn",
    "orderbook": f"{GOLDSKY_BASE_URL}/orderbook-subgraph/0.0.1/gn",
    "open_interest": f"{GOLDSKY_BASE_URL}/oi-subgraph/0.0.6/gn",
}


async def query(session, endpoint, q, variables):
    """Execute GraphQL query."""
    try:
        async with session.post(
            endpoint,
            json={"query": q, "variables": variables},
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("data", {})
    except Exception as e:
        print(f"  Error: {e}")
    return {}


async def test_orderbook_pagination():
    """Test how far orderbook pagination goes."""
    print("\n=== ORDERBOOK PAGINATION LIMITS ===")

    q = """
    query($first: Int!, $skip: Int!) {
        orderFilledEvents(first: $first, skip: $skip, orderBy: timestamp, orderDirection: desc) {
            id
            maker
            taker
        }
    }
    """

    async with aiohttp.ClientSession() as session:
        total_trades = 0
        wallets = set()

        for skip in [0, 100000, 200000, 300000, 400000, 500000]:
            data = await query(session, ENDPOINTS["orderbook"], q, {"first": 1000, "skip": skip})
            events = data.get("orderFilledEvents", [])

            for e in events:
                wallets.add(e.get("maker", "").lower())
                wallets.add(e.get("taker", "").lower())

            print(f"  skip={skip:,}: {len(events)} events, {len(wallets):,} unique wallets")

            if not events:
                print(f"  -> EMPTY at skip={skip:,}")
                break

            total_trades += len(events)
            await asyncio.sleep(0.2)


async def test_pnl_userpositions():
    """Test userPositions from PnL subgraph for unique wallets."""
    print("\n=== PNL SUBGRAPH - userPositions ===")

    q = """
    query($first: Int!, $skip: Int!) {
        userPositions(first: $first, skip: $skip) {
            user
            tokenId
            amount
        }
    }
    """

    async with aiohttp.ClientSession() as session:
        wallets = set()

        for skip in [0, 100000, 200000, 300000, 400000, 500000]:
            data = await query(session, ENDPOINTS["pnl"], q, {"first": 1000, "skip": skip})
            positions = data.get("userPositions", [])

            for p in positions:
                w = p.get("user", "").lower()
                if w:
                    wallets.add(w)

            print(f"  skip={skip:,}: {len(positions)} positions, {len(wallets):,} unique wallets")

            if not positions:
                break

            await asyncio.sleep(0.2)


async def test_activity_subgraph():
    """Test activity subgraph for redemptions, splits, merges."""
    print("\n=== ACTIVITY SUBGRAPH ===")

    # Test redemptions
    q_redemptions = """
    query($first: Int!, $skip: Int!) {
        redemptions(first: $first, skip: $skip, orderBy: timestamp, orderDirection: desc) {
            redeemer
            timestamp
            payout
        }
    }
    """

    # Test splits
    q_splits = """
    query($first: Int!, $skip: Int!) {
        splits(first: $first, skip: $skip, orderBy: timestamp, orderDirection: desc) {
            stakeholder
            timestamp
            amount
        }
    }
    """

    # Test merges
    q_merges = """
    query($first: Int!, $skip: Int!) {
        merges(first: $first, skip: $skip, orderBy: timestamp, orderDirection: desc) {
            stakeholder
            timestamp
            amount
        }
    }
    """

    async with aiohttp.ClientSession() as session:
        wallets_by_type = defaultdict(set)

        for name, q, field in [
            ("redemptions", q_redemptions, "redeemer"),
            ("splits", q_splits, "stakeholder"),
            ("merges", q_merges, "stakeholder"),
        ]:
            print(f"\n  {name.upper()}:")
            entity_name = name

            for skip in [0, 100000, 200000, 300000]:
                data = await query(session, ENDPOINTS["activity"], q, {"first": 1000, "skip": skip})
                items = data.get(entity_name, [])

                for item in items:
                    w = item.get(field, "").lower()
                    if w:
                        wallets_by_type[name].add(w)

                print(f"    skip={skip:,}: {len(items)} items, {len(wallets_by_type[name]):,} unique wallets")

                if not items:
                    break

                await asyncio.sleep(0.2)

        # Combine all activity wallets
        all_activity = set()
        for w in wallets_by_type.values():
            all_activity.update(w)
        print(f"\n  TOTAL ACTIVITY WALLETS: {len(all_activity):,}")


async def test_positions_subgraph():
    """Test positions subgraph directly."""
    print("\n=== POSITIONS SUBGRAPH ===")

    q = """
    query($first: Int!, $skip: Int!) {
        positions(first: $first, skip: $skip) {
            id
            user
            condition
        }
    }
    """

    async with aiohttp.ClientSession() as session:
        wallets = set()

        for skip in [0, 100000, 200000]:
            data = await query(session, ENDPOINTS["positions"], q, {"first": 1000, "skip": skip})
            positions = data.get("positions", [])

            for p in positions:
                w = p.get("user", "").lower()
                if w:
                    wallets.add(w)

            print(f"  skip={skip:,}: {len(positions)} positions, {len(wallets):,} unique wallets")

            if not positions:
                break

            await asyncio.sleep(0.2)


async def test_combine_all_sources():
    """Combine wallets from all available subgraph sources."""
    print("\n=== COMBINING ALL SOURCES ===")

    queries = [
        # Orderbook - trades
        ("orderbook", """
            query($first: Int!, $skip: Int!) {
                orderFilledEvents(first: $first, skip: $skip, orderBy: timestamp, orderDirection: desc) {
                    maker
                    taker
                }
            }
        """, ["maker", "taker"], "orderFilledEvents"),

        # PnL - positions
        ("pnl", """
            query($first: Int!, $skip: Int!) {
                userPositions(first: $first, skip: $skip) {
                    user
                }
            }
        """, ["user"], "userPositions"),

        # Activity - redemptions
        ("activity", """
            query($first: Int!, $skip: Int!) {
                redemptions(first: $first, skip: $skip) {
                    redeemer
                }
            }
        """, ["redeemer"], "redemptions"),

        # Activity - splits
        ("activity", """
            query($first: Int!, $skip: Int!) {
                splits(first: $first, skip: $skip) {
                    stakeholder
                }
            }
        """, ["stakeholder"], "splits"),
    ]

    all_wallets = set()
    source_stats = {}

    async with aiohttp.ClientSession() as session:
        for source, q, fields, entity_name in queries:
            print(f"\n  Fetching from {source}/{entity_name}...")
            source_wallets = set()

            skip = 0
            while skip < 500000:
                data = await query(session, ENDPOINTS[source], q, {"first": 1000, "skip": skip})
                items = data.get(entity_name, [])

                if not items:
                    break

                for item in items:
                    for field in fields:
                        w = item.get(field, "").lower()
                        if w and len(w) == 42:  # Valid ETH address
                            source_wallets.add(w)

                skip += 1000

                # Only fetch 100k from each source for this test
                if skip >= 100000:
                    break

                await asyncio.sleep(0.1)

            source_stats[f"{source}/{entity_name}"] = len(source_wallets)
            all_wallets.update(source_wallets)
            print(f"    -> {len(source_wallets):,} wallets")

    print(f"\n  SOURCE BREAKDOWN:")
    for src, count in sorted(source_stats.items(), key=lambda x: -x[1]):
        print(f"    {src}: {count:,}")

    print(f"\n  TOTAL UNIQUE WALLETS: {len(all_wallets):,}")
    return all_wallets


async def main():
    print("=" * 60)
    print("EXPLORING GOLDSKY SUBGRAPH WALLET SOURCES")
    print("=" * 60)

    start = time.time()

    # Test each source
    await test_orderbook_pagination()
    await test_pnl_userpositions()
    await test_activity_subgraph()
    await test_positions_subgraph()

    # Combine all
    all_wallets = await test_combine_all_sources()

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"TOTAL TIME: {elapsed:.1f}s")
    print(f"TOTAL UNIQUE WALLETS FOUND: {len(all_wallets):,}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
