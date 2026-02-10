"""
Fresh scan test - AGGRESSIVE FETCH
1. Fetch as many trades as possible (target 300k+ events)
2. Calculate 20 core metrics - report rate
3. Calculate 117 metrics - report rate
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import aiohttp
import numpy as np
import time
from collections import defaultdict

# Goldsky API
GOLDSKY_PROJECT_ID = "project_cl6mb8i9h0003e201j6li0diw"
GOLDSKY_BASE_URL = f"https://api.goldsky.com/api/public/{GOLDSKY_PROJECT_ID}/subgraphs"
ORDERBOOK_ENDPOINT = f"{GOLDSKY_BASE_URL}/orderbook-subgraph/0.0.1/gn"


async def query(session, q, variables, retries=3):
    """Execute GraphQL query."""
    for attempt in range(retries):
        try:
            async with session.post(
                ORDERBOOK_ENDPOINT,
                json={"query": q, "variables": variables},
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if "errors" not in data:
                        return data.get("data", {})
        except Exception:
            if attempt < retries - 1:
                await asyncio.sleep(0.5)
    return {}


async def fetch_all_trades(max_events=400000):
    """Fetch trades aggressively - as many as possible."""
    print(f"\n{'='*60}")
    print(f"PHASE 1: AGGRESSIVE FETCH (target {max_events:,} events)")
    print(f"{'='*60}")

    q = """query($first: Int!, $skip: Int!) {
        orderFilledEvents(first: $first, skip: $skip, orderBy: timestamp, orderDirection: desc) {
            timestamp
            maker
            taker
            makerAmountFilled
            takerAmountFilled
            fee
            makerAssetId
        }
    }"""

    trades_by_wallet = defaultdict(list)
    total_events = 0
    start_time = time.time()

    connector = aiohttp.TCPConnector(limit=50, force_close=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        skip = 0
        empty_count = 0

        while total_events < max_events and empty_count < 3:
            data = await query(session, q, {"first": 1000, "skip": skip})
            events = data.get("orderFilledEvents", [])

            if not events:
                empty_count += 1
                skip += 1000
                continue

            empty_count = 0

            for e in events:
                ts = int(e.get("timestamp", 0) or 0)
                maker = e.get("maker", "").lower()
                taker = e.get("taker", "").lower()
                maker_amt = float(e.get("makerAmountFilled", 0) or 0) / 1e6
                taker_amt = float(e.get("takerAmountFilled", 0) or 0) / 1e6
                fee = float(e.get("fee", 0) or 0) / 1e6
                market = e.get("makerAssetId", "")[:20] if e.get("makerAssetId") else "0"

                if maker and len(maker) == 42 and maker_amt > 0:
                    trades_by_wallet[maker].append({
                        "ts": ts,
                        "price": min(1.0, taker_amt / maker_amt) if maker_amt > 0 else 0.5,
                        "size": maker_amt,
                        "value": maker_amt,
                        "side": 1,
                        "fee": fee / 2,
                        "market": market,
                    })

                if taker and len(taker) == 42 and taker_amt > 0:
                    trades_by_wallet[taker].append({
                        "ts": ts,
                        "price": min(1.0, maker_amt / taker_amt) if taker_amt > 0 else 0.5,
                        "size": taker_amt,
                        "value": taker_amt,
                        "side": 0,
                        "fee": fee / 2,
                        "market": market,
                    })

                total_events += 1

            skip += 1000

            if skip % 10000 == 0:
                elapsed = time.time() - start_time
                rate = total_events / elapsed if elapsed > 0 else 0
                print(f"  {total_events:,} events, {len(trades_by_wallet):,} wallets ({rate:,.0f} events/sec)")

    elapsed = time.time() - start_time
    print(f"\nFetch complete:")
    print(f"  Duration: {elapsed:.1f}s")
    print(f"  Total events: {total_events:,}")
    print(f"  Unique wallets: {len(trades_by_wallet):,}")
    print(f"  Rate: {total_events/elapsed:,.0f} events/sec")

    return trades_by_wallet


def prepare_wallet_arrays(trades_by_wallet):
    """Convert trades to numpy arrays for each wallet."""
    print(f"\n{'='*60}")
    print("PHASE 2: PREPARING DATA")
    print(f"{'='*60}")

    wallet_data = {}

    for wallet, trades in trades_by_wallet.items():
        if len(trades) < 2:
            continue

        trades = sorted(trades, key=lambda x: x["ts"])
        n = len(trades)

        markets = list(set(t["market"] for t in trades))
        market_map = {m: i for i, m in enumerate(markets)}
        n_markets = len(markets)

        wallet_data[wallet] = {
            "prices": np.array([t["price"] for t in trades], dtype=np.float64),
            "sizes": np.array([t["size"] for t in trades], dtype=np.float64),
            "fees": np.array([t["fee"] for t in trades], dtype=np.float64),
            "sides": np.array([t["side"] for t in trades], dtype=np.int64),
            "sides_i8": np.array([t["side"] for t in trades], dtype=np.int8),
            "timestamps": np.array([t["ts"] for t in trades], dtype=np.int64),
            "timestamps_f64": np.array([t["ts"] for t in trades], dtype=np.float64),
            "values": np.array([t["value"] for t in trades], dtype=np.float64),
            "market_ids": np.array([market_map[t["market"]] for t in trades], dtype=np.int64),
            "n_markets": n_markets,
            "n_trades": n,
        }

    print(f"  Wallets with 2+ trades: {len(wallet_data):,}")
    return wallet_data


def calculate_20_metrics(wallet_data):
    """Calculate 20 core metrics for all wallets."""
    from src.core.analyzer.fast_core_metrics import calculate_core_20

    print(f"\n{'='*60}")
    print("PHASE 3: CALCULATE 20 CORE METRICS")
    print(f"{'='*60}")

    # Warmup
    print("  Warming up Numba...")
    for w in list(wallet_data.values())[:5]:
        _ = calculate_core_20(
            w["prices"], w["sizes"], w["fees"], w["sides_i8"],
            w["timestamps_f64"], w["values"], w["market_ids"],
            w["n_markets"]
        )

    n_wallets = len(wallet_data)
    print(f"  Calculating for {n_wallets:,} wallets...")

    start = time.perf_counter()
    for addr, w in wallet_data.items():
        _ = calculate_core_20(
            w["prices"], w["sizes"], w["fees"], w["sides_i8"],
            w["timestamps_f64"], w["values"], w["market_ids"],
            w["n_markets"]
        )

    elapsed = time.perf_counter() - start
    rate = n_wallets / elapsed if elapsed > 0 else 0

    print(f"\n  RESULTS:")
    print(f"    Wallets: {n_wallets:,}")
    print(f"    Time: {elapsed:.4f}s")
    print(f"    Rate: {rate:,.0f} wallets/sec")

    return rate


def calculate_117_metrics(wallet_data):
    """Calculate all 117 metrics for all wallets."""
    from src.core.analyzer.numba_metrics import calculate_all_117_metrics

    print(f"\n{'='*60}")
    print("PHASE 4: CALCULATE 117 METRICS")
    print(f"{'='*60}")

    current_ts = int(time.time())

    # Warmup
    print("  Warming up Numba...")
    for w in list(wallet_data.values())[:5]:
        _ = calculate_all_117_metrics(
            w["prices"], w["sizes"], w["fees"], w["sides"],
            w["timestamps"], w["market_ids"], w["values"],
            w["n_markets"], current_ts
        )

    n_wallets = len(wallet_data)
    print(f"  Calculating for {n_wallets:,} wallets...")

    start = time.perf_counter()
    for addr, w in wallet_data.items():
        _ = calculate_all_117_metrics(
            w["prices"], w["sizes"], w["fees"], w["sides"],
            w["timestamps"], w["market_ids"], w["values"],
            w["n_markets"], current_ts
        )

    elapsed = time.perf_counter() - start
    rate = n_wallets / elapsed if elapsed > 0 else 0

    print(f"\n  RESULTS:")
    print(f"    Wallets: {n_wallets:,}")
    print(f"    Time: {elapsed:.4f}s")
    print(f"    Rate: {rate:,.0f} wallets/sec")

    return rate


async def main():
    print("=" * 60)
    print("FRESH SCAN - AGGRESSIVE FETCH")
    print("=" * 60)

    total_start = time.time()

    # Phase 1: Fetch aggressively
    trades_by_wallet = await fetch_all_trades(max_events=400000)

    # Phase 2: Prepare
    wallet_data = prepare_wallet_arrays(trades_by_wallet)

    if not wallet_data:
        print("\nNo wallets with enough trades found!")
        return

    # Phase 3: 20 metrics
    rate_20 = calculate_20_metrics(wallet_data)

    # Phase 4: 117 metrics
    rate_117 = calculate_117_metrics(wallet_data)

    # Summary
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total wallets scanned: {len(trades_by_wallet):,}")
    print(f"Wallets with metrics: {len(wallet_data):,}")
    print(f"")
    print(f"20 CORE METRICS:  {rate_20:,.0f} wallets/sec")
    print(f"117 METRICS:      {rate_117:,.0f} wallets/sec")
    print(f"")
    print(f"Total time: {total_time:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
