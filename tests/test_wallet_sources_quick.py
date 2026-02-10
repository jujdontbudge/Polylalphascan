"""
Quick test of wallet sources from different Goldsky subgraphs.
"""
import asyncio
import aiohttp
import time

GOLDSKY_PROJECT_ID = "project_cl6mb8i9h0003e201j6li0diw"
GOLDSKY_BASE_URL = f"https://api.goldsky.com/api/public/{GOLDSKY_PROJECT_ID}/subgraphs"

ENDPOINTS = {
    "activity": f"{GOLDSKY_BASE_URL}/activity-subgraph/0.0.4/gn",
    "pnl": f"{GOLDSKY_BASE_URL}/pnl-subgraph/0.0.14/gn",
    "orderbook": f"{GOLDSKY_BASE_URL}/orderbook-subgraph/0.0.1/gn",
}


async def query(session, endpoint, q, variables):
    try:
        async with session.post(
            endpoint,
            json={"query": q, "variables": variables},
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=20),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("data", {})
    except Exception as e:
        print(f"Error: {e}")
    return {}


async def main():
    print("Testing wallet sources...\n")
    start = time.time()

    all_wallets = set()

    async with aiohttp.ClientSession() as session:
        # 1. Orderbook trades (what we currently use)
        print("1. Orderbook (trades):")
        q = """query($first: Int!, $skip: Int!) {
            orderFilledEvents(first: $first, skip: $skip) { maker taker }
        }"""

        orderbook_wallets = set()
        for skip in range(0, 50000, 1000):
            data = await query(session, ENDPOINTS["orderbook"], q, {"first": 1000, "skip": skip})
            events = data.get("orderFilledEvents", [])
            if not events:
                break
            for e in events:
                orderbook_wallets.add(e.get("maker", "").lower())
                orderbook_wallets.add(e.get("taker", "").lower())
            print(f"  skip={skip}: +{len(events)} events, total wallets: {len(orderbook_wallets)}")
        print(f"  -> {len(orderbook_wallets):,} wallets from orderbook\n")
        all_wallets.update(orderbook_wallets)

        # 2. PnL userPositions
        print("2. PnL (userPositions):")
        q = """query($first: Int!, $skip: Int!) {
            userPositions(first: $first, skip: $skip) { user }
        }"""

        pnl_wallets = set()
        for skip in range(0, 50000, 1000):
            data = await query(session, ENDPOINTS["pnl"], q, {"first": 1000, "skip": skip})
            positions = data.get("userPositions", [])
            if not positions:
                break
            for p in positions:
                w = p.get("user", "").lower()
                if w:
                    pnl_wallets.add(w)
            print(f"  skip={skip}: +{len(positions)} positions, total wallets: {len(pnl_wallets)}")
        print(f"  -> {len(pnl_wallets):,} wallets from pnl\n")

        new_from_pnl = pnl_wallets - orderbook_wallets
        print(f"  -> {len(new_from_pnl):,} NEW wallets (not in orderbook)\n")
        all_wallets.update(pnl_wallets)

        # 3. Activity redemptions
        print("3. Activity (redemptions):")
        q = """query($first: Int!, $skip: Int!) {
            redemptions(first: $first, skip: $skip) { redeemer }
        }"""

        activity_wallets = set()
        for skip in range(0, 50000, 1000):
            data = await query(session, ENDPOINTS["activity"], q, {"first": 1000, "skip": skip})
            items = data.get("redemptions", [])
            if not items:
                break
            for item in items:
                w = item.get("redeemer", "").lower()
                if w:
                    activity_wallets.add(w)
            print(f"  skip={skip}: +{len(items)} redemptions, total wallets: {len(activity_wallets)}")
        print(f"  -> {len(activity_wallets):,} wallets from activity\n")

        new_from_activity = activity_wallets - all_wallets
        print(f"  -> {len(new_from_activity):,} NEW wallets (not in orderbook+pnl)\n")
        all_wallets.update(activity_wallets)

    elapsed = time.time() - start
    print("=" * 50)
    print(f"TOTAL UNIQUE WALLETS: {len(all_wallets):,}")
    print(f"Time: {elapsed:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
