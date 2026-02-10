"""API clients for Polymarket endpoints"""
from .base_client import BaseClient
from .gamma_client import GammaClient
from .data_client import DataClient
from .clob_client import CLOBClient
from .polygon_rpc_client import (
    PolygonRPCClient,
    ProviderPool,
    RPCProvider,
    OrderFilledEvent,
    CTF_EXCHANGE_MULTI,
    CTF_EXCHANGE_LEGACY,
)
from .async_data_client import (
    AsyncDataClient,
    AsyncRateLimiter,
    fetch_wallets_parallel,
)

# High-performance data sources (Phase 3)
try:
    from .goldsky_client import GoldskyClient, fetch_batch_from_goldsky
    GOLDSKY_AVAILABLE = True
except ImportError:
    GOLDSKY_AVAILABLE = False

try:
    from .hypersync_client import HyperSyncClient, discover_wallets_hypersync
    HYPERSYNC_AVAILABLE = True
except ImportError:
    HYPERSYNC_AVAILABLE = False

try:
    from .unified_data_router import UnifiedDataRouter, fetch_wallets_unified
    UNIFIED_ROUTER_AVAILABLE = True
except ImportError:
    UNIFIED_ROUTER_AVAILABLE = False

__all__ = [
    "BaseClient",
    "GammaClient",
    "DataClient",
    "CLOBClient",
    "PolygonRPCClient",
    "ProviderPool",
    "RPCProvider",
    "OrderFilledEvent",
    "CTF_EXCHANGE_MULTI",
    "CTF_EXCHANGE_LEGACY",
    # Async clients
    "AsyncDataClient",
    "AsyncRateLimiter",
    "fetch_wallets_parallel",
    # High-performance data sources
    "GOLDSKY_AVAILABLE",
    "HYPERSYNC_AVAILABLE",
    "UNIFIED_ROUTER_AVAILABLE",
]

# Conditionally add to __all__
if GOLDSKY_AVAILABLE:
    __all__.extend(["GoldskyClient", "fetch_batch_from_goldsky"])

if HYPERSYNC_AVAILABLE:
    __all__.extend(["HyperSyncClient", "discover_wallets_hypersync"])

if UNIFIED_ROUTER_AVAILABLE:
    __all__.extend(["UnifiedDataRouter", "fetch_wallets_unified"])
