"""Caching layer for metrics and wallet data"""
from .redis_cache import MetricCache, get_cache
from .trade_cache import TradeCache, get_trade_cache, CachedTrade, WalletSyncInfo, TradeCacheStats
from .binary_trade_cache import BinaryTradeCache, BinaryCacheStats, migrate_sqlite_to_binary

__all__ = [
    # Redis/metric cache
    "MetricCache",
    "get_cache",
    # Trade cache (local SQLite)
    "TradeCache",
    "get_trade_cache",
    "CachedTrade",
    "WalletSyncInfo",
    "TradeCacheStats",
    # Binary trade cache (ultra-fast NumPy)
    "BinaryTradeCache",
    "BinaryCacheStats",
    "migrate_sqlite_to_binary",
]
