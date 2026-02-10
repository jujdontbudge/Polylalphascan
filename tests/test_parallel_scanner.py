"""Tests for parallel deep scanner and async client"""
import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime

from src.api.async_data_client import (
    AsyncDataClient,
    AsyncRateLimiter,
    Trade,
    Position,
    WalletActivity,
)
from src.core.analyzer.parallel_deep_scanner import (
    ParallelDeepScanner,
    ParallelScanResult,
    calculate_metrics_for_wallet,
)


class TestAsyncRateLimiter:
    """Tests for async rate limiter"""

    def test_rate_limiter_allows_burst(self):
        """Rate limiter allows burst requests up to burst_size"""
        import asyncio

        async def run_test():
            limiter = AsyncRateLimiter(requests_per_second=10.0, burst_size=5)

            # Should allow 5 requests immediately (burst)
            wait_times = []
            for _ in range(5):
                wait = await limiter.acquire()
                wait_times.append(wait)

            # All burst requests should have minimal wait
            assert all(w < 0.1 for w in wait_times)

        asyncio.run(run_test())

    def test_rate_limiter_tracks_stats(self):
        """Rate limiter tracks request stats"""
        import asyncio

        async def run_test():
            limiter = AsyncRateLimiter(requests_per_second=100.0, burst_size=10)

            for _ in range(5):
                await limiter.acquire()

            assert limiter.total_requests == 5

        asyncio.run(run_test())


class TestTradeFromDict:
    """Tests for Trade.from_dict"""

    def test_trade_from_api_response(self):
        """Trade parses API response correctly"""
        data = {
            "id": "trade_123",
            "transactionHash": "0xabc",
            "timestamp": "2024-01-15T10:30:00Z",
            "marketId": "market_1",
            "conditionId": "cond_1",
            "proxyWallet": "0xwallet",
            "side": "BUY",
            "outcome": "Yes",
            "price": 0.65,
            "size": 100,
            "usdcSize": 65.0,
            "fee": 0.5,
        }

        trade = Trade.from_dict(data)

        assert trade.id == "trade_123"
        assert trade.wallet_address == "0xwallet"
        assert trade.side == "BUY"
        assert trade.price == 0.65
        assert trade.size == 100
        assert trade.value == 65.0

    def test_trade_calculates_value_from_price_size(self):
        """Trade calculates value when not provided"""
        data = {
            "id": "trade_1",
            "price": 0.50,
            "size": 200,
        }

        trade = Trade.from_dict(data)
        assert trade.value == 100.0


class TestPositionFromDict:
    """Tests for Position.from_dict"""

    def test_position_from_api_response(self):
        """Position parses API response correctly"""
        data = {
            "marketId": "market_1",
            "conditionId": "cond_1",
            "user": "0xwallet",
            "outcome": "Yes",
            "outcomeIndex": 0,
            "size": 100,
            "avgPrice": 0.40,
            "currentPrice": 0.60,
            "realizedPnl": 5.0,
        }

        position = Position.from_dict(data)

        assert position.market_id == "market_1"
        assert position.size == 100
        assert position.avg_price == 0.40
        assert position.current_price == 0.60
        assert position.unrealized_pnl == pytest.approx(20.0, abs=0.01)  # (0.60 - 0.40) * 100


class TestWalletActivity:
    """Tests for WalletActivity"""

    def test_calculate_summary(self):
        """WalletActivity calculates summary correctly"""
        activity = WalletActivity(address="0xtest")
        activity.trades = [
            Trade(
                id="1", transaction_hash="", timestamp=datetime(2024, 1, 1),
                market_id="m1", condition_id="c1", wallet_address="0xtest",
                side="BUY", outcome="Yes", price=0.50, size=100, value=50.0, fee=0.1
            ),
            Trade(
                id="2", transaction_hash="", timestamp=datetime(2024, 1, 2),
                market_id="m1", condition_id="c1", wallet_address="0xtest",
                side="SELL", outcome="Yes", price=0.70, size=100, value=70.0, fee=0.1
            ),
        ]

        activity.calculate_summary()

        assert activity.total_trades == 2
        assert activity.total_volume == 120.0
        assert activity.first_trade == datetime(2024, 1, 1)
        assert activity.last_trade == datetime(2024, 1, 2)


class TestCalculateMetricsForWallet:
    """Tests for standalone metric calculation function"""

    def test_calculates_metrics_for_valid_data(self):
        """Function calculates metrics for valid trade data"""
        trades_data = [
            {
                "id": "1",
                "timestamp": "2024-01-15T10:00:00Z",
                "conditionId": "cond_1",
                "side": "BUY",
                "price": 0.40,
                "size": 100,
                "usdcSize": 40.0,
                "fee": 0.1,
            },
            {
                "id": "2",
                "timestamp": "2024-01-16T10:00:00Z",
                "conditionId": "cond_1",
                "side": "SELL",
                "price": 0.70,
                "size": 100,
                "usdcSize": 70.0,
                "fee": 0.1,
            },
        ]

        address, result = calculate_metrics_for_wallet(
            "0xtest",
            trades_data,
            []  # No positions
        )

        assert address == "0xtest"
        assert result is not None
        assert result["total_calculated"] > 0
        assert result["trade_count"] == 2
        assert "classification" in result

    def test_returns_none_for_empty_trades(self):
        """Function returns None for empty trades"""
        address, result = calculate_metrics_for_wallet("0xtest", [], [])

        assert address == "0xtest"
        assert result is None


class TestParallelScanResult:
    """Tests for ParallelScanResult"""

    def test_wallets_per_second(self):
        """Calculates wallets per second correctly"""
        result = ParallelScanResult()
        result.wallets_scanned = 100
        result.duration_seconds = 10.0

        assert result.wallets_per_second == 10.0

    def test_success_rate(self):
        """Calculates success rate correctly"""
        result = ParallelScanResult()
        result.wallets_scanned = 80
        result.wallets_failed = 20

        assert result.success_rate == 80.0


class TestParallelDeepScanner:
    """Tests for ParallelDeepScanner"""

    def test_initialization(self):
        """Scanner initializes with correct defaults"""
        scanner = ParallelDeepScanner()

        assert scanner.max_concurrent_http == 200
        assert scanner.requests_per_second == 100.0
        assert scanner.db_batch_size == 1000

    def test_custom_configuration(self):
        """Scanner accepts custom configuration"""
        scanner = ParallelDeepScanner(
            max_concurrent_http=100,
            requests_per_second=50.0,
            db_batch_size=500,
        )

        assert scanner.max_concurrent_http == 100
        assert scanner.requests_per_second == 50.0
        assert scanner.db_batch_size == 500


class TestBatchMetricRepository:
    """Tests for batch metric operations"""

    def test_batch_set_wallet_metrics_structure(self):
        """Verify batch data structure is correct"""
        # This tests the data structure without needing DB
        wallet_metrics = [
            {
                "wallet_id": 1,
                "metrics": {1: 100.5, 2: 50.0, 3: None},  # None should be filtered
                "time_period": "all_time",
            },
            {
                "wallet_id": 2,
                "metrics": {1: 200.0, 2: 75.0},
                "time_period": "all_time",
            },
        ]

        # Count expected records (excluding None values)
        expected_records = 0
        for entry in wallet_metrics:
            expected_records += len([v for v in entry["metrics"].values() if v is not None])

        assert expected_records == 4  # 2 + 2 valid metrics
