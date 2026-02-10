"""Global configuration for PolyScan Alpha Finder v2.0"""
from pathlib import Path
from dataclasses import dataclass
from typing import List

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"
EXPORTS_DIR = PROJECT_ROOT / "exports"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
EXPORTS_DIR.mkdir(exist_ok=True)

# Database
DATABASE_PATH = DATA_DIR / "polyscan.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"


@dataclass
class APIConfig:
    """API endpoint configuration"""
    # Base URLs
    GAMMA_API_BASE = "https://gamma-api.polymarket.com"
    DATA_API_BASE = "https://data-api.polymarket.com"
    CLOB_API_BASE = "https://clob.polymarket.com"

    # Rate limits (requests per 10 seconds)
    DATA_API_LIMIT = 200
    POSITIONS_API_LIMIT = 150
    GAMMA_API_LIMIT = 300

    # Delays between requests (seconds)
    DATA_API_DELAY = 0.2
    GAMMA_API_DELAY = 0.1
    DEFAULT_DELAY = 0.2

    # Timeouts (seconds)
    REQUEST_TIMEOUT = 30

    # Pagination
    DEFAULT_PAGE_SIZE = 100
    MAX_PAGE_SIZE = 500


@dataclass
class ScannerConfig:
    """Scanner pipeline configuration"""
    # Watering hole categories for market discovery
    WATERING_HOLE_TAGS: List[str] = None

    # Lite filter thresholds
    MIN_VOLUME_USD = 500.0
    MIN_TRADE_COUNT = 10

    # Deep scan settings
    MAX_WALLETS_PER_SCAN = 1000
    BATCH_SIZE = 50

    # Ranking settings
    TOP_N = 100
    BUFFER_N = 150  # Internal buffer for efficiency

    def __post_init__(self):
        if self.WATERING_HOLE_TAGS is None:
            # All major Polymarket categories with active algorithmic traders
            self.WATERING_HOLE_TAGS = [
                # Original watering holes
                "crypto",
                "weather",
                "economics",
                # Politics - high volume, many bots
                "politics",
                "elections",
                "trump",
                "biden",
                # Sports - fast-moving, requires bots
                "sports",
                "nfl",
                "nba",
                "soccer",
                "mma",
                # Finance & Business
                "finance",
                "earnings",
                "stocks",
                "fed",
                # Other high-activity categories
                "entertainment",
                "ai",
                "tech",
            ]


@dataclass
class ClassifierConfig:
    """Bot/Quant classification thresholds"""
    # Quant score thresholds
    BOT_THRESHOLD = 80
    QUANT_THRESHOLD = 50
    HUMAN_THRESHOLD = 20

    # Classification labels
    LABEL_BOT = "bot"
    LABEL_QUANT = "quant"
    LABEL_HUMAN = "human"
    LABEL_UNKNOWN = "unknown"


@dataclass
class ContinuousMonitorConfig:
    """Continuous trade monitoring configuration for zero-missed-wallets"""
    POLL_INTERVAL: float = 5.0          # Seconds between polls
    TRADES_PER_POLL: int = 500          # Trades to fetch per request
    CHECKPOINT_KEY: str = "last_trade_id"
    MAX_RETRY_DELAY: int = 60           # Max backoff on errors

    # Categories for discovery/filtering (informational)
    TRACKED_CATEGORIES: List[str] = None

    def __post_init__(self):
        if self.TRACKED_CATEGORIES is None:
            self.TRACKED_CATEGORIES = [
                "sports",
                "earnings",
                "crypto",
                "finance",
                "weather",
                "culture",
                "politics",
                "elections",
                "tech",
            ]


@dataclass
class MetricsConfig:
    """Metrics calculation configuration"""
    # Total metrics
    TOTAL_METRICS = 117
    TOTAL_GROUPS = 15

    # Group ranges
    GROUPS = {
        'A': {'name': 'Profitability', 'start': 1, 'end': 10, 'count': 10},
        'B': {'name': 'Risk & Stability', 'start': 11, 'end': 21, 'count': 11},
        'C': {'name': 'Win/Loss Dynamics', 'start': 22, 'end': 32, 'count': 11},
        'D': {'name': 'Activity & Frequency', 'start': 33, 'end': 40, 'count': 8},
        'E': {'name': 'Timing & Speed', 'start': 41, 'end': 48, 'count': 8},
        'F': {'name': 'Money Management', 'start': 49, 'end': 57, 'count': 9},
        'G': {'name': 'Strategy Signature', 'start': 58, 'end': 66, 'count': 9},
        'H': {'name': 'Psychological', 'start': 67, 'end': 74, 'count': 8},
        'I': {'name': 'Advanced Alpha', 'start': 75, 'end': 83, 'count': 9},
        'J': {'name': 'Hygiene & Metadata', 'start': 84, 'end': 94, 'count': 11},
        'K': {'name': 'Copy-ability', 'start': 95, 'end': 100, 'count': 6},
        'L': {'name': 'Market Intelligence', 'start': 101, 'end': 104, 'count': 4},
        'M': {'name': 'Execution Quality', 'start': 105, 'end': 108, 'count': 4},
        'N': {'name': 'Bot/Quant Confirmation', 'start': 109, 'end': 113, 'count': 5},
        'O': {'name': 'Portfolio-Level', 'start': 114, 'end': 117, 'count': 4},
    }


# Global instances
api_config = APIConfig()
scanner_config = ScannerConfig()
classifier_config = ClassifierConfig()
metrics_config = MetricsConfig()
continuous_config = ContinuousMonitorConfig()


@dataclass
class RealtimeConfig:
    """Real-time trade capture configuration for WebSocket + REST backup"""

    # WebSocket
    RTDS_WS_URL: str = "wss://ws-live-data.polymarket.com"
    WS_PING_INTERVAL: float = 30.0
    WS_RECONNECT_MAX: float = 60.0

    # REST backup
    REST_POLL_INTERVAL: float = 10.0
    REST_OVERLAP_SECONDS: int = 30

    # Gap detection
    GAP_CHECK_INTERVAL: float = 60.0

    # Checkpoints
    CHECKPOINT_INTERVAL: float = 10.0
    CHECKPOINT_OVERLAP: int = 60  # Seconds to re-fetch on restart

    # Deduplication
    BLOOM_CAPACITY: int = 1_000_000
    BLOOM_ERROR_RATE: float = 0.001
    RECENT_CACHE_SIZE: int = 50_000


realtime_config = RealtimeConfig()


@dataclass
class RPCConfig:
    """Polygon RPC configuration for direct blockchain queries"""

    # Provider configuration (no API key required)
    # Note: Free tier limits vary. Check provider docs for current limits.
    PROVIDERS = [
        {
            "name": "1RPC",
            "url": "https://1rpc.io/matic",
            "rate_limit": 10.0,
            "max_block_range": 5000,
            "priority": 5,
        },
        {
            "name": "Polygon",
            "url": "https://polygon-rpc.com",
            "rate_limit": 5.0,
            "max_block_range": 1000,
            "priority": 4,
        },
        {
            "name": "PublicNode",
            "url": "https://polygon-bor-rpc.publicnode.com",
            "rate_limit": 10.0,
            "max_block_range": 5000,
            "priority": 3,
        },
        {
            "name": "Chainstack",
            "url": "https://polygon-mainnet.g.alchemy.com/v2/demo",
            "rate_limit": 5.0,
            "max_block_range": 2000,
            "priority": 2,
        },
        {
            "name": "DrpcPublic",
            "url": "https://polygon.drpc.org",
            "rate_limit": 10.0,
            "max_block_range": 5000,
            "priority": 1,
        },
    ]

    # Harvesting settings
    MAX_CONCURRENT: int = 100
    REQUEST_TIMEOUT: int = 30

    # Polymarket CTF Exchange deployment (approximate block)
    POLYMARKET_START_BLOCK: int = 30_000_000

    # Contract addresses
    CTF_EXCHANGE_MULTI: str = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
    CTF_EXCHANGE_LEGACY: str = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"

    # Polygon block time (~2 seconds)
    BLOCK_TIME_SECONDS: float = 2.0

    @property
    def blocks_per_hour(self) -> int:
        return int(3600 / self.BLOCK_TIME_SECONDS)

    @property
    def blocks_per_day(self) -> int:
        return self.blocks_per_hour * 24

    @property
    def combined_rate_limit(self) -> float:
        """Total requests per second across all providers"""
        return sum(p["rate_limit"] for p in self.PROVIDERS)


rpc_config = RPCConfig()
