"""
Benchmark: Calculate ALL 117 metrics - get real rate.
"""
import numpy as np
import time
import sys
sys.path.insert(0, '.')

from src.core.analyzer.numba_metrics import calculate_all_117_metrics
from src.core.analyzer.fast_core_metrics import calculate_core_20

def generate_fake_wallet(n_trades=50, n_markets=5):
    """Generate fake wallet data for benchmarking."""
    return {
        "prices": np.random.uniform(0.1, 0.9, n_trades).astype(np.float64),
        "sizes": np.random.uniform(10, 1000, n_trades).astype(np.float64),
        "fees": np.random.uniform(0.1, 5, n_trades).astype(np.float64),
        "sides": np.random.randint(0, 2, n_trades).astype(np.int64),
        "timestamps": np.sort(np.random.randint(1700000000, 1710000000, n_trades)).astype(np.int64),
        "market_ids": np.random.randint(0, n_markets, n_trades).astype(np.int64),
        "values": np.random.uniform(100, 10000, n_trades).astype(np.float64),
        "n_markets": n_markets,
    }

def test_117_metrics():
    """Test 117 metric calculation rate."""
    print("=" * 60)
    print("117 METRICS BENCHMARK")
    print("=" * 60)

    # Generate test wallets
    n_wallets = 10000
    print(f"\nGenerating {n_wallets:,} fake wallets...")

    wallets = []
    for i in range(n_wallets):
        n_trades = np.random.randint(10, 200)
        n_markets = np.random.randint(2, 20)
        wallets.append(generate_fake_wallet(n_trades, n_markets))

    current_ts = int(time.time())

    # Warmup (compile Numba)
    print("Warming up Numba JIT...")
    for w in wallets[:10]:
        _ = calculate_all_117_metrics(
            w["prices"], w["sizes"], w["fees"], w["sides"],
            w["timestamps"], w["market_ids"], w["values"],
            w["n_markets"], current_ts
        )

    # Benchmark 117 metrics
    print(f"\nCalculating 117 metrics for {n_wallets:,} wallets...")
    start = time.perf_counter()

    for w in wallets:
        _ = calculate_all_117_metrics(
            w["prices"], w["sizes"], w["fees"], w["sides"],
            w["timestamps"], w["market_ids"], w["values"],
            w["n_markets"], current_ts
        )

    elapsed = time.perf_counter() - start
    rate_117 = n_wallets / elapsed

    print(f"\n117 METRICS:")
    print(f"  Time: {elapsed:.4f}s")
    print(f"  Rate: {rate_117:,.0f} wallets/sec")

    return rate_117

def test_20_metrics():
    """Test 20 core metric calculation rate."""
    print("\n" + "=" * 60)
    print("20 CORE METRICS BENCHMARK")
    print("=" * 60)

    # Generate test wallets
    n_wallets = 10000
    print(f"\nGenerating {n_wallets:,} fake wallets...")

    wallets = []
    for i in range(n_wallets):
        n_trades = np.random.randint(10, 200)
        n_markets = np.random.randint(2, 20)
        w = generate_fake_wallet(n_trades, n_markets)
        # Convert for 20-metric function
        w["sides_i8"] = w["sides"].astype(np.int8)
        w["timestamps_f64"] = w["timestamps"].astype(np.float64)
        wallets.append(w)

    # Warmup
    print("Warming up Numba JIT...")
    for w in wallets[:10]:
        _ = calculate_core_20(
            w["prices"], w["sizes"], w["fees"], w["sides_i8"],
            w["timestamps_f64"], w["values"], w["market_ids"],
            w["n_markets"]
        )

    # Benchmark
    print(f"\nCalculating 20 core metrics for {n_wallets:,} wallets...")
    start = time.perf_counter()

    for w in wallets:
        _ = calculate_core_20(
            w["prices"], w["sizes"], w["fees"], w["sides_i8"],
            w["timestamps_f64"], w["values"], w["market_ids"],
            w["n_markets"]
        )

    elapsed = time.perf_counter() - start
    rate_20 = n_wallets / elapsed

    print(f"\n20 CORE METRICS:")
    print(f"  Time: {elapsed:.4f}s")
    print(f"  Rate: {rate_20:,.0f} wallets/sec")

    return rate_20

if __name__ == "__main__":
    rate_117 = test_117_metrics()
    rate_20 = test_20_metrics()

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"117 metrics: {rate_117:,.0f} wallets/sec")
    print(f"20 metrics:  {rate_20:,.0f} wallets/sec")
    print(f"Speedup (20 vs 117): {rate_20/rate_117:.1f}x faster")
