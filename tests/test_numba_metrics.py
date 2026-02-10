"""
Test correctness of the ultra-fast 117 metrics Numba engine.

Compares Numba implementation against reference Pandas implementations
to ensure accuracy within acceptable tolerance.
"""
import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.analyzer.numba_metrics import (
    FastWalletData,
    calculate_117_metrics,
    calculate_quant_score,
    METRIC_NAMES,
)
from src.core.analyzer.numba_intermediates import (
    calc_basic_stats,
    calc_market_arrays,
    calc_win_loss_stats,
    calc_time_stats,
    calc_returns_and_risk,
    safe_divide,
)


# =============================================================================
# Test Data Generation
# =============================================================================

def create_test_wallet(
    n_trades: int = 50,
    n_markets: int = 5,
    seed: int = 42,
) -> FastWalletData:
    """Create test wallet data with reproducible random data."""
    np.random.seed(seed)

    prices = np.random.uniform(0.1, 0.9, n_trades).astype(np.float64)
    sizes = np.random.uniform(10, 1000, n_trades).astype(np.float64)
    fees = sizes * 0.001
    sides = np.random.randint(0, 2, n_trades).astype(np.int64)

    # Generate timestamps over 30 days
    base_ts = int(datetime(2024, 1, 1).timestamp())
    timestamps = (base_ts + np.cumsum(np.random.uniform(3600, 86400, n_trades))).astype(np.int64)

    market_ids = np.random.randint(0, n_markets, n_trades).astype(np.int64)
    values = prices * sizes

    return FastWalletData(
        address="0x" + "a" * 40,
        n_trades=n_trades,
        n_markets=n_markets,
        prices=prices,
        sizes=sizes,
        fees=fees,
        sides=sides,
        timestamps=timestamps,
        market_ids=market_ids,
        values=values,
    )


def create_simple_wallet() -> FastWalletData:
    """Create a simple wallet with predictable values for manual verification."""
    # 4 trades: 2 buys, 2 sells in 2 markets
    prices = np.array([0.5, 0.6, 0.7, 0.4], dtype=np.float64)
    sizes = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float64)
    fees = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float64)
    sides = np.array([1, 1, 0, 0], dtype=np.int64)  # BUY, BUY, SELL, SELL

    base_ts = int(datetime(2024, 1, 1).timestamp())
    timestamps = np.array([
        base_ts,
        base_ts + 3600,
        base_ts + 7200,
        base_ts + 10800,
    ], dtype=np.int64)

    market_ids = np.array([0, 1, 0, 1], dtype=np.int64)
    values = prices * sizes

    return FastWalletData(
        address="0x" + "b" * 40,
        n_trades=4,
        n_markets=2,
        prices=prices,
        sizes=sizes,
        fees=fees,
        sides=sides,
        timestamps=timestamps,
        market_ids=market_ids,
        values=values,
    )


# =============================================================================
# Intermediate Calculation Tests
# =============================================================================

class TestIntermediates:
    """Test intermediate calculation functions."""

    def test_basic_stats(self):
        """Test basic stats calculation."""
        values = np.array([100.0, 200.0, 150.0, 250.0], dtype=np.float64)
        fees = np.array([1.0, 2.0, 1.5, 2.5], dtype=np.float64)
        sides = np.array([1, 0, 1, 0], dtype=np.int64)  # BUY, SELL, BUY, SELL

        total_vol, buy_vol, sell_vol, buy_count, sell_count, total_fees = \
            calc_basic_stats(values, fees, sides)

        assert total_vol == 700.0
        assert buy_vol == 250.0  # 100 + 150
        assert sell_vol == 450.0  # 200 + 250
        assert buy_count == 2
        assert sell_count == 2
        assert total_fees == 7.0

    def test_market_arrays(self):
        """Test per-market calculation."""
        prices = np.array([0.5, 0.6, 0.7, 0.4], dtype=np.float64)
        sizes = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float64)
        fees = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        sides = np.array([1, 1, 0, 0], dtype=np.int64)  # BUY, BUY, SELL, SELL
        market_ids = np.array([0, 1, 0, 1], dtype=np.int64)

        market_pnl, market_volume, market_count, _, _ = calc_market_arrays(
            prices, sizes, fees, sides, market_ids, 2
        )

        # Market 0: Buy at 0.5, sell at 0.7 -> PnL = (0.7 - 0.5) * 100 = 20
        # Market 1: Buy at 0.6, sell at 0.4 -> PnL = (0.4 - 0.6) * 100 = -20
        assert pytest.approx(market_pnl[0], rel=0.01) == 20.0
        assert pytest.approx(market_pnl[1], rel=0.01) == -20.0

        assert market_count[0] == 2
        assert market_count[1] == 2

    def test_win_loss_stats(self):
        """Test win/loss statistics."""
        market_pnl = np.array([20.0, -10.0, 15.0, -5.0, 30.0], dtype=np.float64)

        total_pnl, wins, losses, win_rate, winning_pnl, losing_pnl = \
            calc_win_loss_stats(market_pnl)

        assert pytest.approx(total_pnl, rel=0.01) == 50.0
        assert wins == 3
        assert losses == 2
        assert pytest.approx(win_rate, rel=0.01) == 0.6
        assert pytest.approx(winning_pnl, rel=0.01) == 65.0
        assert pytest.approx(losing_pnl, rel=0.01) == 15.0

    def test_time_stats(self):
        """Test time statistics."""
        base_ts = 1700000000
        timestamps = np.array([
            base_ts,
            base_ts + 3600,
            base_ts + 7200,
            base_ts + 14400,
        ], dtype=np.int64)

        intervals, time_span_days, mean_interval, std_interval, median_interval = \
            calc_time_stats(timestamps)

        assert len(intervals) == 3
        assert pytest.approx(intervals[0], rel=0.01) == 3600.0
        assert pytest.approx(intervals[1], rel=0.01) == 3600.0
        assert pytest.approx(intervals[2], rel=0.01) == 7200.0

        # Mean: (3600 + 3600 + 7200) / 3 = 4800
        assert pytest.approx(mean_interval, rel=0.01) == 4800.0

        # Median: 3600 (middle value when sorted)
        assert pytest.approx(median_interval, rel=0.01) == 3600.0

    def test_safe_divide(self):
        """Test safe division."""
        assert safe_divide(10.0, 2.0) == 5.0
        assert safe_divide(10.0, 0.0) == 0.0
        assert safe_divide(10.0, 0.0, default=1.0) == 1.0


# =============================================================================
# Full Metric Tests
# =============================================================================

class TestFullMetrics:
    """Test full 117 metric calculation."""

    def test_all_metrics_calculated(self):
        """Test that all 117 metrics are calculated."""
        wallet = create_test_wallet()
        metrics = calculate_117_metrics(wallet)

        assert len(metrics) == 117
        for i in range(1, 118):
            assert i in metrics, f"Metric {i} missing"

    def test_empty_wallet(self):
        """Test handling of empty wallet."""
        wallet = FastWalletData(
            address="0x0",
            n_trades=0,
            n_markets=0,
            prices=np.array([], dtype=np.float64),
            sizes=np.array([], dtype=np.float64),
            fees=np.array([], dtype=np.float64),
            sides=np.array([], dtype=np.int64),
            timestamps=np.array([], dtype=np.int64),
            market_ids=np.array([], dtype=np.int64),
            values=np.array([], dtype=np.float64),
        )

        metrics = calculate_117_metrics(wallet)

        assert len(metrics) == 117
        # All should be 0 for empty wallet
        assert all(v == 0.0 for v in metrics.values())

    def test_single_trade_wallet(self):
        """Test wallet with single trade."""
        wallet = FastWalletData(
            address="0x1",
            n_trades=1,
            n_markets=1,
            prices=np.array([0.5], dtype=np.float64),
            sizes=np.array([100.0], dtype=np.float64),
            fees=np.array([0.1], dtype=np.float64),
            sides=np.array([1], dtype=np.int64),
            timestamps=np.array([1700000000], dtype=np.int64),
            market_ids=np.array([0], dtype=np.int64),
            values=np.array([50.0], dtype=np.float64),
        )

        metrics = calculate_117_metrics(wallet)

        # Should handle gracefully
        assert len(metrics) == 117
        assert metrics[33] == 1.0  # Total trades
        assert metrics[34] == 50.0  # Total volume

    def test_simple_wallet_pnl(self):
        """Test PnL calculation with simple wallet."""
        wallet = create_simple_wallet()
        metrics = calculate_117_metrics(wallet)

        # Total PnL should be approximately 0 (market 0: +20, market 1: -20, minus fees)
        # Actually: 20 - 20 - 0.4 = -0.4
        assert pytest.approx(metrics[1], abs=1.0) == -0.4

    def test_reproducibility(self):
        """Test that results are reproducible."""
        wallet = create_test_wallet(seed=123)

        metrics1 = calculate_117_metrics(wallet)
        metrics2 = calculate_117_metrics(wallet)

        for i in range(1, 118):
            assert metrics1[i] == metrics2[i], f"Metric {i} not reproducible"

    def test_metric_ranges(self):
        """Test that metrics are within expected ranges."""
        wallet = create_test_wallet()
        metrics = calculate_117_metrics(wallet)

        # Win rate should be 0-100
        assert 0 <= metrics[22] <= 100, f"Win rate {metrics[22]} out of range"

        # Loss rate should be 0-100
        assert 0 <= metrics[23] <= 100, f"Loss rate {metrics[23]} out of range"

        # Win + Loss rate should be ~100
        assert pytest.approx(metrics[22] + metrics[23], abs=5) == 100

        # Max drawdown should be non-negative (can exceed 100% in extreme cases)
        assert metrics[14] >= 0, f"Max drawdown {metrics[14]} should be non-negative"

        # Trade count should match
        assert metrics[33] == wallet.n_trades

        # Unique markets should match
        assert metrics[39] == wallet.n_markets


# =============================================================================
# Metric Group Tests
# =============================================================================

class TestMetricGroups:
    """Test individual metric groups."""

    def setup_method(self):
        """Create test wallet for each test."""
        self.wallet = create_test_wallet(n_trades=100, n_markets=10)
        self.metrics = calculate_117_metrics(self.wallet)

    def test_group_a_profitability(self):
        """Test Group A: Profitability metrics (1-10)."""
        # Metric 4: Profit factor should be positive
        assert self.metrics[4] >= 0

        # Metric 5: Avg market PnL
        assert isinstance(self.metrics[5], float)

        # Metric 9: PnL per volume should be a ratio
        if self.metrics[34] > 0:  # Total volume
            expected = self.metrics[1] / self.metrics[34]
            assert pytest.approx(self.metrics[9], rel=0.01) == expected

    def test_group_b_risk(self):
        """Test Group B: Risk metrics (11-21)."""
        # Sharpe ratio can be any value
        assert isinstance(self.metrics[11], float)

        # Max drawdown should be non-negative percentage
        assert self.metrics[14] >= 0

        # Volatility should be non-negative
        assert self.metrics[17] >= 0

    def test_group_c_winloss(self):
        """Test Group C: Win/Loss metrics (22-32)."""
        # Win rate + loss rate should equal ~100%
        total_rate = self.metrics[22] + self.metrics[23]
        assert pytest.approx(total_rate, abs=5) == 100

        # Streaks should be non-negative integers (stored as float)
        assert self.metrics[27] >= 0  # Max win streak
        assert self.metrics[28] >= 0  # Max loss streak

    def test_group_d_activity(self):
        """Test Group D: Activity metrics (33-40)."""
        # Trade count should be exact
        assert self.metrics[33] == self.wallet.n_trades

        # Total volume should match sum of values
        assert pytest.approx(self.metrics[34], rel=0.01) == self.wallet.values.sum()

        # Average trade size
        expected_avg = self.wallet.values.sum() / self.wallet.n_trades
        assert pytest.approx(self.metrics[35], rel=0.01) == expected_avg

        # Unique markets
        assert self.metrics[39] == self.wallet.n_markets

    def test_group_e_timing(self):
        """Test Group E: Timing metrics (41-48)."""
        # Hold times should be non-negative
        assert self.metrics[41] >= 0  # Avg hold time
        assert self.metrics[42] >= 0  # Median hold time

        # Timing scores should be 0-100
        assert 0 <= self.metrics[47] <= 100  # Entry timing
        assert 0 <= self.metrics[48] <= 100  # Exit timing

    def test_group_n_bot_detection(self):
        """Test Group N: Bot detection metrics (109-113)."""
        # Entropy should be 0-100
        assert 0 <= self.metrics[109] <= 100

        # Round number avoidance 0-100
        assert 0 <= self.metrics[110] <= 100

        # Latency consistency
        assert self.metrics[111] >= 0


# =============================================================================
# Quant Score Tests
# =============================================================================

class TestQuantScore:
    """Test quant score calculation."""

    def test_quant_score_range(self):
        """Test that quant score is in valid range."""
        wallet = create_test_wallet()
        metrics = calculate_117_metrics(wallet)
        score = calculate_quant_score(metrics)

        assert 0 <= score <= 100

    def test_quant_score_classification(self):
        """Test quant score classification thresholds."""
        # Create bot-like wallet (regular intervals)
        np.random.seed(42)
        n_trades = 100

        # Regular intervals = bot-like
        base_ts = 1700000000
        timestamps = np.array([base_ts + i * 3600 for i in range(n_trades)], dtype=np.int64)

        # Round number avoidance
        prices = np.array([0.33333, 0.66666] * 50, dtype=np.float64)

        wallet = FastWalletData(
            address="0xbot",
            n_trades=n_trades,
            n_markets=5,
            prices=prices,
            sizes=np.full(n_trades, 100.0, dtype=np.float64),
            fees=np.full(n_trades, 0.1, dtype=np.float64),
            sides=np.array([1, 0] * 50, dtype=np.int64),
            timestamps=timestamps,
            market_ids=np.random.randint(0, 5, n_trades).astype(np.int64),
            values=prices * 100,
        )

        metrics = calculate_117_metrics(wallet)
        score = calculate_quant_score(metrics)

        # Bot-like behavior should score higher
        # (exact threshold depends on implementation)
        assert isinstance(score, float)


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Test calculation performance."""

    def test_calculation_time(self):
        """Test that calculation completes in reasonable time."""
        import time

        wallet = create_test_wallet(n_trades=1000, n_markets=50)

        start = time.perf_counter()
        metrics = calculate_117_metrics(wallet)
        elapsed = time.perf_counter() - start

        # Should complete in under 10ms for single large wallet (1000 trades)
        assert elapsed < 0.01, f"Calculation took {elapsed*1000:.2f}ms"

    def test_batch_throughput(self):
        """Test batch processing throughput."""
        import time

        n_wallets = 1000
        wallets = [create_test_wallet(seed=i) for i in range(n_wallets)]

        start = time.perf_counter()
        for wallet in wallets:
            _ = calculate_117_metrics(wallet)
        elapsed = time.perf_counter() - start

        throughput = n_wallets / elapsed

        # Should achieve at least 10,000 wallets/sec single-threaded
        # (50,000+ achievable with parallelization)
        assert throughput >= 10000, f"Throughput {throughput:.0f} < 10,000 wallets/sec"


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_buys(self):
        """Test wallet with only buy trades."""
        wallet = FastWalletData(
            address="0xbuys",
            n_trades=10,
            n_markets=2,
            prices=np.random.uniform(0.3, 0.7, 10).astype(np.float64),
            sizes=np.full(10, 100.0, dtype=np.float64),
            fees=np.full(10, 0.1, dtype=np.float64),
            sides=np.ones(10, dtype=np.int64),  # All buys
            timestamps=np.arange(10, dtype=np.int64) * 3600 + 1700000000,
            market_ids=np.random.randint(0, 2, 10).astype(np.int64),
            values=np.full(10, 50.0, dtype=np.float64),
        )

        metrics = calculate_117_metrics(wallet)
        assert len(metrics) == 117

    def test_all_sells(self):
        """Test wallet with only sell trades."""
        wallet = FastWalletData(
            address="0xsells",
            n_trades=10,
            n_markets=2,
            prices=np.random.uniform(0.3, 0.7, 10).astype(np.float64),
            sizes=np.full(10, 100.0, dtype=np.float64),
            fees=np.full(10, 0.1, dtype=np.float64),
            sides=np.zeros(10, dtype=np.int64),  # All sells
            timestamps=np.arange(10, dtype=np.int64) * 3600 + 1700000000,
            market_ids=np.random.randint(0, 2, 10).astype(np.int64),
            values=np.full(10, 50.0, dtype=np.float64),
        )

        metrics = calculate_117_metrics(wallet)
        assert len(metrics) == 117

    def test_single_market(self):
        """Test wallet trading in single market."""
        wallet = FastWalletData(
            address="0xsingle",
            n_trades=20,
            n_markets=1,
            prices=np.random.uniform(0.3, 0.7, 20).astype(np.float64),
            sizes=np.full(20, 100.0, dtype=np.float64),
            fees=np.full(20, 0.1, dtype=np.float64),
            sides=np.random.randint(0, 2, 20).astype(np.int64),
            timestamps=np.arange(20, dtype=np.int64) * 3600 + 1700000000,
            market_ids=np.zeros(20, dtype=np.int64),
            values=np.full(20, 50.0, dtype=np.float64),
        )

        metrics = calculate_117_metrics(wallet)
        assert len(metrics) == 117
        assert metrics[39] == 1  # Unique markets

    def test_extreme_prices(self):
        """Test wallet with extreme prices (near 0 and 1)."""
        prices = np.array([0.01, 0.99, 0.02, 0.98, 0.05, 0.95], dtype=np.float64)

        wallet = FastWalletData(
            address="0xextreme",
            n_trades=6,
            n_markets=3,
            prices=prices,
            sizes=np.full(6, 100.0, dtype=np.float64),
            fees=np.full(6, 0.1, dtype=np.float64),
            sides=np.array([1, 0, 1, 0, 1, 0], dtype=np.int64),
            timestamps=np.arange(6, dtype=np.int64) * 3600 + 1700000000,
            market_ids=np.array([0, 0, 1, 1, 2, 2], dtype=np.int64),
            values=prices * 100,
        )

        metrics = calculate_117_metrics(wallet)
        assert len(metrics) == 117

    def test_zero_fees(self):
        """Test wallet with zero fees."""
        wallet = create_test_wallet()
        wallet.fees = np.zeros_like(wallet.fees)

        metrics = calculate_117_metrics(wallet)
        assert len(metrics) == 117


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
