"""
Full end-to-end benchmark: collect wallets from all sources + calculate metrics.
"""
import asyncio
import aiohttp
import time
import numpy as np
from collections import defaultdict

GOLDSKY_PROJECT_ID = "project_cl6mb8i9h0003e201j6li0diw"
GOLDSKY_BASE_URL = f"https://api.goldsky.com/api/public/{GOLDSKY_PROJECT_ID}/subgraphs"

ENDPOINTS = {
    "activity": f"{GOLDSKY_BASE_URL}/activity-subgraph/0.0.4/gn",
    "orderbook": f"{GOLDSKY_BASE_URL}/orderbook-subgraph/0.0.1/gn",
}


async def query(session, endpoint, q, variables):
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
        pass
    return {}


async def collect_trades_and_wallets(session, max_trades=200000):
    """Collect trades from orderbook, return trades grouped by wallet."""
    q = """query($first: Int!, $skip: Int!) {
        orderFilledEvents(first: $first, skip: $skip, orderBy: timestamp, orderDirection: desc) {
            timestamp
            maker
            taker
            makerAmountFilled
            takerAmountFilled
            fee
        }
    }"""

    trades_by_wallet = defaultdict(list)
    skip = 0

    while skip < max_trades:
        data = await query(session, ENDPOINTS["orderbook"], q, {"first": 1000, "skip": skip})
        events = data.get("orderFilledEvents", [])
        if not events:
            break

        for e in events:
            maker = e.get("maker", "").lower()
            taker = e.get("taker", "").lower()

            # Parse trade data
            ts = int(e.get("timestamp", 0) or 0)
            maker_amt = float(e.get("makerAmountFilled", 0) or 0) / 1e6
            taker_amt = float(e.get("takerAmountFilled", 0) or 0) / 1e6
            fee = float(e.get("fee", 0) or 0) / 1e6

            if maker and len(maker) == 42 and maker_amt > 0:
                trades_by_wallet[maker].append({
                    "ts": ts,
                    "value": maker_amt,
                    "side": 1,  # maker giving, so selling
                    "fee": fee / 2,
                })

            if taker and len(taker) == 42 and taker_amt > 0:
                trades_by_wallet[taker].append({
                    "ts": ts,
                    "value": taker_amt,
                    "side": 0,  # taker receiving, so buying
                    "fee": fee / 2,
                })

        skip += 1000
        if skip % 10000 == 0:
            print(f"  Fetched {skip:,} events, {len(trades_by_wallet):,} wallets...")

    return trades_by_wallet


async def collect_activity_wallets(session, max_items=200000):
    """Get additional wallets from activity (redemptions)."""
    q = """query($first: Int!, $skip: Int!) {
        redemptions(first: $first, skip: $skip) { redeemer }
    }"""

    wallets = set()
    skip = 0

    while skip < max_items:
        data = await query(session, ENDPOINTS["activity"], q, {"first": 1000, "skip": skip})
        items = data.get("redemptions", [])
        if not items:
            break

        for item in items:
            w = item.get("redeemer", "").lower()
            if w and len(w) == 42:
                wallets.add(w)

        skip += 1000
        if skip % 50000 == 0:
            print(f"  Fetched {skip:,} redemptions, {len(wallets):,} wallets...")

    return wallets


def calculate_metrics_fast(trades: list) -> dict:
    """Calculate key metrics for a wallet's trades."""
    if len(trades) < 2:
        return None

    # Sort by timestamp
    trades = sorted(trades, key=lambda x: x["ts"])

    # Convert to numpy
    n = len(trades)
    values = np.array([t["value"] for t in trades], dtype=np.float64)
    sides = np.array([t["side"] for t in trades], dtype=np.int8)
    fees = np.array([t["fee"] for t in trades], dtype=np.float64)
    timestamps = np.array([t["ts"] for t in trades], dtype=np.float64)

    # Basic stats
    total_volume = values.sum()
    total_fees = fees.sum()
    buy_volume = values[sides == 0].sum()
    sell_volume = values[sides == 1].sum()

    # PnL
    total_pnl = sell_volume - buy_volume - total_fees

    # ROI
    roi = (total_pnl / buy_volume * 100) if buy_volume > 0 else 0.0

    # Time span (days)
    time_span = max(1, (timestamps[-1] - timestamps[0]) / 86400)

    # Trades per day
    trades_per_day = n / time_span

    # Simple returns for volatility
    if n > 2:
        returns = np.diff(values) / np.maximum(values[:-1], 0.001)
        volatility = np.std(returns) * 100
        mean_return = np.mean(returns)

        # Sharpe (simplified)
        if volatility > 0:
            sharpe = (mean_return * np.sqrt(252)) / (volatility / 100)
        else:
            sharpe = 0.0
    else:
        volatility = 0.0
        sharpe = 0.0

    # Drawdown
    cum_pnl = np.cumsum(np.where(sides == 1, values, -values))
    peak = np.maximum.accumulate(cum_pnl)
    drawdown = peak - cum_pnl
    max_dd = drawdown.max()
    max_dd_pct = (max_dd / peak.max() * 100) if peak.max() > 0 else 0.0

    return {
        "total_pnl": total_pnl,
        "roi_pct": roi,
        "sharpe": sharpe,
        "max_dd_pct": max_dd_pct,
        "total_volume": total_volume,
        "trade_count": n,
        "trades_per_day": trades_per_day,
        "volatility_pct": volatility,
    }


async def main():
    print("=" * 60)
    print("FULL BENCHMARK: COLLECT + CALCULATE")
    print("=" * 60)

    start_total = time.time()

    connector = aiohttp.TCPConnector(limit=30, force_close=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Phase 1: Collect trades
        print("\n1. COLLECTING TRADES FROM ORDERBOOK...")
        start = time.time()
        trades_by_wallet = await collect_trades_and_wallets(session, max_trades=300000)
        orderbook_time = time.time() - start
        orderbook_wallets = set(trades_by_wallet.keys())
        total_trades = sum(len(t) for t in trades_by_wallet.values())
        print(f"   {len(orderbook_wallets):,} wallets with {total_trades:,} trades in {orderbook_time:.1f}s")

        # Phase 2: Get additional wallets from activity
        print("\n2. COLLECTING WALLETS FROM ACTIVITY...")
        start = time.time()
        activity_wallets = await collect_activity_wallets(session, max_items=200000)
        activity_time = time.time() - start
        print(f"   {len(activity_wallets):,} wallets from redemptions in {activity_time:.1f}s")

        # Combine
        all_wallets = orderbook_wallets | activity_wallets
        new_from_activity = activity_wallets - orderbook_wallets
        print(f"\n   TOTAL UNIQUE WALLETS: {len(all_wallets):,}")
        print(f"   ({len(new_from_activity):,} new from activity)")

    # Phase 3: Calculate metrics
    print("\n3. CALCULATING METRICS...")
    start = time.time()

    wallets_with_metrics = 0
    wallets_processed = 0

    for wallet, trades in trades_by_wallet.items():
        metrics = calculate_metrics_fast(trades)
        if metrics:
            wallets_with_metrics += 1
        wallets_processed += 1

    calc_time = time.time() - start
    rate = wallets_processed / calc_time if calc_time > 0 else 0

    print(f"   Processed: {wallets_processed:,} wallets")
    print(f"   With metrics: {wallets_with_metrics:,}")
    print(f"   Time: {calc_time:.4f}s")
    print(f"   RATE: {rate:,.0f} wallets/sec")

    # Summary
    total_time = time.time() - start_total
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total wallets discovered: {len(all_wallets):,}")
    print(f"Wallets with trade data: {len(orderbook_wallets):,}")
    print(f"Metrics calculated: {wallets_with_metrics:,}")
    print(f"Calculation rate: {rate:,.0f} wallets/sec")
    print(f"Total time: {total_time:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
