"""
Trade Deduplicator - Multi-layer deduplication for high throughput.

Uses:
1. Bloom filter for fast "probably new" checks
2. LRU cache for exact recent trade checks
3. Database unique constraint as final guarantee
"""
import logging
from collections import OrderedDict
from datetime import datetime
from typing import Optional

from config.settings import realtime_config

logger = logging.getLogger(__name__)

try:
    from pybloom_live import BloomFilter
    BLOOM_AVAILABLE = True
except ImportError:
    BLOOM_AVAILABLE = False
    logger.warning("pybloom_live not installed, using set-based deduplication")


class TradeDeduplicator:
    """Multi-layer deduplication for high throughput trade capture"""

    def __init__(
        self,
        bloom_capacity: int = None,
        bloom_error_rate: float = None,
        cache_size: int = None,
    ):
        """
        Initialize deduplicator.

        Args:
            bloom_capacity: Bloom filter capacity (default from config)
            bloom_error_rate: Bloom filter error rate (default from config)
            cache_size: LRU cache size (default from config)
        """
        self.bloom_capacity = bloom_capacity or realtime_config.BLOOM_CAPACITY
        self.bloom_error_rate = bloom_error_rate or realtime_config.BLOOM_ERROR_RATE
        self.cache_size = cache_size or realtime_config.RECENT_CACHE_SIZE

        # Validate config
        if self.bloom_capacity <= 0:
            raise ValueError(f"bloom_capacity must be > 0, got {self.bloom_capacity}")
        if not (0 < self.bloom_error_rate < 1):
            raise ValueError(f"bloom_error_rate must be between 0 and 1 (exclusive), got {self.bloom_error_rate}")

        logger.info(
            f"Deduplicator init: bloom_capacity={self.bloom_capacity}, "
            f"bloom_error_rate={self.bloom_error_rate}, cache_size={self.cache_size}"
        )

        # Layer 1: Bloom filter (fast "probably new" check)
        if BLOOM_AVAILABLE:
            self.bloom = BloomFilter(
                capacity=self.bloom_capacity,
                error_rate=self.bloom_error_rate,
            )
        else:
            # Fallback to simple set (uses more memory but exact)
            self.bloom = set()

        # Layer 2: LRU cache (exact check for recent trades)
        self.recent: OrderedDict[str, datetime] = OrderedDict()

        # Stats
        self.total_checked = 0
        self.duplicates_found = 0
        self.bloom_positives = 0

    def is_duplicate(self, trade_id: str) -> bool:
        """
        Check if trade already processed.

        Args:
            trade_id: Trade identifier (transaction hash)

        Returns:
            True if duplicate, False if new
        """
        self.total_checked += 1

        # Fast path: bloom filter says "definitely new"
        if BLOOM_AVAILABLE:
            if trade_id not in self.bloom:
                return False
            self.bloom_positives += 1
        else:
            if trade_id not in self.bloom:
                return False

        # Slow path: check exact cache
        if trade_id in self.recent:
            self.duplicates_found += 1
            return True

        # Bloom says "maybe" but LRU says "no" — could be a bloom false positive
        # or a true positive that was evicted from LRU. We treat it as new (false
        # negative risk) since the DB unique constraint is the final safety layer.
        # To reduce this risk, ensure cache_size >= expected unique trade volume.
        return False

    def record(self, trade_id: str) -> None:
        """
        Record trade as processed.

        Args:
            trade_id: Trade identifier (transaction hash)
        """
        # Add to bloom filter
        if BLOOM_AVAILABLE:
            self.bloom.add(trade_id)
        else:
            self.bloom.add(trade_id)

        # Add to recent cache
        self.recent[trade_id] = datetime.utcnow()

        # Evict old entries to maintain cache size
        while len(self.recent) > self.cache_size:
            self.recent.popitem(last=False)

    def check_and_record(self, trade_id: str) -> bool:
        """
        Check if duplicate and record if new.

        Args:
            trade_id: Trade identifier

        Returns:
            True if new (recorded), False if duplicate (skipped)
        """
        if self.is_duplicate(trade_id):
            return False
        self.record(trade_id)
        return True

    def get_stats(self) -> dict:
        """Get deduplication statistics"""
        return {
            "total_checked": self.total_checked,
            "duplicates_found": self.duplicates_found,
            "bloom_positives": self.bloom_positives,
            "cache_size": len(self.recent),
            "bloom_capacity": self.bloom_capacity,
            "duplicate_rate": (
                self.duplicates_found / self.total_checked * 100
                if self.total_checked > 0 else 0
            ),
        }

    def clear(self) -> None:
        """Clear all deduplication state"""
        if BLOOM_AVAILABLE:
            self.bloom = BloomFilter(
                capacity=self.bloom_capacity,
                error_rate=self.bloom_error_rate,
            )
        else:
            self.bloom = set()
        self.recent.clear()
        self.total_checked = 0
        self.duplicates_found = 0
        self.bloom_positives = 0
