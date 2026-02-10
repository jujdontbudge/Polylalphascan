"""Tests for trade deduplicator"""
import pytest
from datetime import datetime

from src.core.realtime.deduplicator import TradeDeduplicator


class TestTradeDeduplicator:
    """Tests for TradeDeduplicator"""

    def test_new_trade_not_duplicate(self):
        """Test that new trades are not marked as duplicates"""
        dedup = TradeDeduplicator(bloom_capacity=1000, cache_size=100)

        assert dedup.is_duplicate("0xabc123") == False
        assert dedup.is_duplicate("0xdef456") == False

    def test_recorded_trade_is_duplicate(self):
        """Test that recorded trades are marked as duplicates"""
        dedup = TradeDeduplicator(bloom_capacity=1000, cache_size=100)

        dedup.record("0xabc123")
        assert dedup.is_duplicate("0xabc123") == True

    def test_check_and_record(self):
        """Test check_and_record returns True for new, False for duplicate"""
        dedup = TradeDeduplicator(bloom_capacity=1000, cache_size=100)

        # First time - should return True (new)
        assert dedup.check_and_record("0xtrade1") == True

        # Second time - should return False (duplicate)
        assert dedup.check_and_record("0xtrade1") == False

    def test_multiple_trades(self):
        """Test multiple different trades"""
        dedup = TradeDeduplicator(bloom_capacity=1000, cache_size=100)

        trades = [f"0xtrade{i}" for i in range(100)]

        # Record all trades
        for trade in trades:
            assert dedup.check_and_record(trade) == True

        # All should now be duplicates
        for trade in trades:
            assert dedup.is_duplicate(trade) == True

    def test_cache_eviction(self):
        """Test that old entries are evicted when cache is full"""
        dedup = TradeDeduplicator(bloom_capacity=1000, cache_size=10)

        # Record more trades than cache size
        for i in range(20):
            dedup.record(f"0xtrade{i}")

        # Cache should only hold 10 entries
        assert len(dedup.recent) == 10

        # Most recent entries should be in cache
        assert "0xtrade19" in dedup.recent
        assert "0xtrade10" in dedup.recent

        # Oldest entries should be evicted from cache
        assert "0xtrade0" not in dedup.recent

    def test_stats(self):
        """Test statistics tracking"""
        dedup = TradeDeduplicator(bloom_capacity=1000, cache_size=100)

        # Check some trades
        dedup.check_and_record("0xtrade1")
        dedup.check_and_record("0xtrade2")
        dedup.check_and_record("0xtrade1")  # Duplicate

        stats = dedup.get_stats()

        assert stats["total_checked"] == 3
        assert stats["duplicates_found"] == 1
        assert stats["cache_size"] == 2

    def test_clear(self):
        """Test clearing deduplication state"""
        dedup = TradeDeduplicator(bloom_capacity=1000, cache_size=100)

        dedup.record("0xtrade1")
        dedup.record("0xtrade2")

        assert len(dedup.recent) == 2

        dedup.clear()

        assert len(dedup.recent) == 0
        assert dedup.total_checked == 0
        assert dedup.duplicates_found == 0

    def test_duplicate_rate_calculation(self):
        """Test duplicate rate calculation"""
        dedup = TradeDeduplicator(bloom_capacity=1000, cache_size=100)

        # 2 unique, 2 duplicates = 50% duplicate rate
        dedup.check_and_record("0xtrade1")  # New
        dedup.check_and_record("0xtrade2")  # New
        dedup.check_and_record("0xtrade1")  # Duplicate
        dedup.check_and_record("0xtrade2")  # Duplicate

        stats = dedup.get_stats()
        assert stats["duplicate_rate"] == 50.0

    def test_empty_stats(self):
        """Test stats with no trades"""
        dedup = TradeDeduplicator()
        stats = dedup.get_stats()

        assert stats["total_checked"] == 0
        assert stats["duplicate_rate"] == 0
