"""
Redis Cache - High-performance caching layer for metrics and wallet data.

Provides:
- 1-hour TTL for metric results
- Hash-based cache invalidation
- Batch get/set operations
- Fallback to in-memory cache if Redis unavailable
"""
import json
import hashlib
import logging
from typing import Dict, Optional, Any, List
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Try to import redis
try:
    import redis
    from redis import Redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis not installed. Install with: pip install redis")


@dataclass
class CacheEntry:
    """Single cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    ttl_seconds: int
    data_hash: str

    def is_expired(self) -> bool:
        """Check if entry has expired"""
        age = (datetime.utcnow() - self.created_at).total_seconds()
        return age > self.ttl_seconds


class InMemoryCache:
    """
    In-memory cache fallback when Redis is unavailable.

    Simple dict-based cache with TTL support.
    """

    def __init__(self, default_ttl: int = 3600):
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        entry = self._cache.get(key)
        if entry is None:
            return None
        if entry.is_expired():
            del self._cache[key]
            return None
        return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        ttl = ttl or self.default_ttl
        data_hash = self._compute_hash(value)

        self._cache[key] = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.utcnow(),
            ttl_seconds=ttl,
            data_hash=data_hash,
        )
        return True

    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def mget(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values"""
        return {key: self.get(key) for key in keys if self.get(key) is not None}

    def mset(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple values"""
        for key, value in mapping.items():
            self.set(key, value, ttl)
        return True

    def clear(self):
        """Clear all cache entries"""
        self._cache.clear()

    def size(self) -> int:
        """Get number of cached entries"""
        return len(self._cache)

    def _compute_hash(self, value: Any) -> str:
        """Compute hash of value for invalidation"""
        json_str = json.dumps(value, sort_keys=True, default=str)
        return hashlib.md5(json_str.encode()).hexdigest()


class MetricCache:
    """
    Redis-based cache for wallet metrics.

    Features:
    - Automatic fallback to in-memory cache
    - Hash-based cache invalidation
    - Batch operations for performance
    - Configurable TTL
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        ttl_seconds: int = 3600,
        prefix: str = "polyscan:",
    ):
        self.ttl = ttl_seconds
        self.prefix = prefix

        self._redis: Optional[Redis] = None
        self._memory_cache = InMemoryCache(default_ttl=ttl_seconds)

        if REDIS_AVAILABLE:
            try:
                self._redis = redis.from_url(redis_url)
                # Test connection
                self._redis.ping()
                logger.info(f"Connected to Redis at {redis_url}")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Using in-memory cache.")
                self._redis = None

    def _make_key(self, key: str) -> str:
        """Make prefixed cache key"""
        return f"{self.prefix}{key}"

    def get_wallet_metrics(
        self,
        address: str,
        time_period: str = "all_time",
    ) -> Optional[Dict[int, float]]:
        """
        Get cached metrics for a wallet.

        Args:
            address: Wallet address
            time_period: Time period (all_time, 7d, 31d)

        Returns:
            Dict of metric_id -> value, or None if not cached
        """
        key = self._make_key(f"metrics:{address.lower()}:{time_period}")

        if self._redis:
            try:
                data = self._redis.get(key)
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.debug(f"Redis get failed: {e}")

        return self._memory_cache.get(key)

    def set_wallet_metrics(
        self,
        address: str,
        metrics: Dict[int, float],
        time_period: str = "all_time",
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Cache metrics for a wallet.

        Args:
            address: Wallet address
            metrics: Dict of metric_id -> value
            time_period: Time period
            ttl: Override default TTL

        Returns:
            True if cached successfully
        """
        key = self._make_key(f"metrics:{address.lower()}:{time_period}")
        ttl = ttl or self.ttl

        # Convert int keys to strings for JSON
        metrics_json = {str(k): v for k, v in metrics.items()}

        if self._redis:
            try:
                self._redis.setex(key, ttl, json.dumps(metrics_json))
                return True
            except Exception as e:
                logger.debug(f"Redis set failed: {e}")

        return self._memory_cache.set(key, metrics_json, ttl)

    def get_wallet_classification(
        self,
        address: str,
    ) -> Optional[Dict[str, Any]]:
        """Get cached classification for a wallet"""
        key = self._make_key(f"class:{address.lower()}")

        if self._redis:
            try:
                data = self._redis.get(key)
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.debug(f"Redis get failed: {e}")

        return self._memory_cache.get(key)

    def set_wallet_classification(
        self,
        address: str,
        classification: str,
        quant_score: float,
        ttl: Optional[int] = None,
    ) -> bool:
        """Cache classification for a wallet"""
        key = self._make_key(f"class:{address.lower()}")
        ttl = ttl or self.ttl

        data = {
            "classification": classification,
            "quant_score": quant_score,
            "cached_at": datetime.utcnow().isoformat(),
        }

        if self._redis:
            try:
                self._redis.setex(key, ttl, json.dumps(data))
                return True
            except Exception as e:
                logger.debug(f"Redis set failed: {e}")

        return self._memory_cache.set(key, data, ttl)

    def invalidate_wallet(self, address: str) -> bool:
        """
        Invalidate all cached data for a wallet.

        Use when wallet data changes.
        """
        patterns = [
            f"metrics:{address.lower()}:*",
            f"class:{address.lower()}",
        ]

        success = True

        if self._redis:
            try:
                for pattern in patterns:
                    full_pattern = self._make_key(pattern)
                    keys = self._redis.keys(full_pattern)
                    if keys:
                        self._redis.delete(*keys)
            except Exception as e:
                logger.debug(f"Redis invalidation failed: {e}")
                success = False

        # Also clear memory cache
        for period in ["all_time", "7d", "31d"]:
            key = self._make_key(f"metrics:{address.lower()}:{period}")
            self._memory_cache.delete(key)

        key = self._make_key(f"class:{address.lower()}")
        self._memory_cache.delete(key)

        return success

    def batch_get_metrics(
        self,
        addresses: List[str],
        time_period: str = "all_time",
    ) -> Dict[str, Optional[Dict[int, float]]]:
        """
        Get metrics for multiple wallets.

        Args:
            addresses: List of wallet addresses
            time_period: Time period

        Returns:
            Dict mapping address -> metrics (or None)
        """
        keys = [
            self._make_key(f"metrics:{addr.lower()}:{time_period}")
            for addr in addresses
        ]

        result = {}

        if self._redis:
            try:
                values = self._redis.mget(keys)
                for addr, value in zip(addresses, values):
                    if value:
                        result[addr] = json.loads(value)
                    else:
                        result[addr] = None
                return result
            except Exception as e:
                logger.debug(f"Redis mget failed: {e}")

        # Fallback to memory cache
        cached = self._memory_cache.mget(keys)
        for addr, key in zip(addresses, keys):
            result[addr] = cached.get(key)

        return result

    def batch_set_metrics(
        self,
        wallet_metrics: Dict[str, Dict[int, float]],
        time_period: str = "all_time",
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Cache metrics for multiple wallets.

        Args:
            wallet_metrics: Dict mapping address -> metrics
            time_period: Time period
            ttl: Override default TTL

        Returns:
            True if all cached successfully
        """
        ttl = ttl or self.ttl

        if self._redis:
            try:
                pipe = self._redis.pipeline()
                for addr, metrics in wallet_metrics.items():
                    key = self._make_key(f"metrics:{addr.lower()}:{time_period}")
                    metrics_json = {str(k): v for k, v in metrics.items()}
                    pipe.setex(key, ttl, json.dumps(metrics_json))
                pipe.execute()
                return True
            except Exception as e:
                logger.debug(f"Redis batch set failed: {e}")

        # Fallback to memory cache
        mapping = {
            self._make_key(f"metrics:{addr.lower()}:{time_period}"): {str(k): v for k, v in metrics.items()}
            for addr, metrics in wallet_metrics.items()
        }
        return self._memory_cache.mset(mapping, ttl)

    def clear(self):
        """Clear all cached data"""
        if self._redis:
            try:
                keys = self._redis.keys(f"{self.prefix}*")
                if keys:
                    self._redis.delete(*keys)
            except Exception as e:
                logger.debug(f"Redis clear failed: {e}")

        self._memory_cache.clear()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            "backend": "redis" if self._redis else "memory",
            "memory_size": self._memory_cache.size(),
        }

        if self._redis:
            try:
                info = self._redis.info("memory")
                stats["redis_used_memory"] = info.get("used_memory_human", "unknown")
                stats["redis_keys"] = self._redis.dbsize()
            except Exception:
                pass

        return stats


# Global cache instance
_cache: Optional[MetricCache] = None


def get_cache(redis_url: str = "redis://localhost:6379") -> MetricCache:
    """Get or create global cache instance"""
    global _cache
    if _cache is None:
        _cache = MetricCache(redis_url=redis_url)
    return _cache
