"""
Prometheus Metrics - Application monitoring and observability.

Provides metrics for:
- Scan throughput and latency
- API request rates
- Cache hit rates
- Worker health

Usage:
    from src.monitoring.metrics import WALLETS_SCANNED, SCAN_DURATION

    WALLETS_SCANNED.inc()
    with SCAN_DURATION.time():
        scan_wallet(...)
"""
import logging
import threading
from typing import Dict, Any, Optional
from functools import wraps
import time

logger = logging.getLogger(__name__)

# Try to import prometheus_client
try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        Summary,
        start_http_server,
        generate_latest,
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed. Install with: pip install prometheus-client")

    # Create dummy metric classes
    class DummyMetric:
        def __init__(self, *args, **kwargs):
            pass

        def inc(self, amount=1):
            pass

        def dec(self, amount=1):
            pass

        def set(self, value):
            pass

        def observe(self, value):
            pass

        def labels(self, *args, **kwargs):
            return self

        def time(self):
            return DummyTimer()

    class DummyTimer:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    Counter = Histogram = Gauge = Summary = DummyMetric

    def start_http_server(port):
        pass

    def generate_latest():
        return b""

    CONTENT_TYPE_LATEST = "text/plain"


# =============================================================================
# Metric Definitions
# =============================================================================

# Scan metrics
WALLETS_SCANNED = Counter(
    "polyscan_wallets_scanned_total",
    "Total number of wallets scanned",
    ["status"],  # success, failed, no_trades
)

SCAN_DURATION = Histogram(
    "polyscan_scan_duration_seconds",
    "Time spent scanning wallets",
    ["phase"],  # fetch, calc, save
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
)

THROUGHPUT = Gauge(
    "polyscan_throughput_wallets_per_second",
    "Current scanning throughput",
)

METRICS_CALCULATED = Counter(
    "polyscan_metrics_calculated_total",
    "Total number of metrics calculated",
)

METRICS_SAVED = Counter(
    "polyscan_metrics_saved_total",
    "Total number of metrics saved to database",
)

# API metrics
API_REQUESTS = Counter(
    "polyscan_api_requests_total",
    "Total API requests",
    ["source", "status"],  # source: rest, goldsky, hypersync; status: success, error
)

API_LATENCY = Histogram(
    "polyscan_api_latency_seconds",
    "API request latency",
    ["source"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

# Cache metrics
CACHE_HITS = Counter(
    "polyscan_cache_hits_total",
    "Cache hits",
    ["cache_type"],  # metrics, classification, trades
)

CACHE_MISSES = Counter(
    "polyscan_cache_misses_total",
    "Cache misses",
    ["cache_type"],
)

CACHE_SIZE = Gauge(
    "polyscan_cache_size",
    "Current cache size",
    ["cache_type"],
)

# Database metrics
DB_QUERIES = Counter(
    "polyscan_db_queries_total",
    "Database queries",
    ["operation"],  # select, insert, update
)

DB_LATENCY = Histogram(
    "polyscan_db_latency_seconds",
    "Database query latency",
    ["operation"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

# Worker metrics
ACTIVE_WORKERS = Gauge(
    "polyscan_active_workers",
    "Number of active workers",
)

PENDING_TASKS = Gauge(
    "polyscan_pending_tasks",
    "Number of pending tasks",
)

# Classification metrics
CLASSIFICATIONS = Counter(
    "polyscan_classifications_total",
    "Wallet classifications",
    ["classification"],  # bot, quant, human, unknown
)


# =============================================================================
# Helper Functions
# =============================================================================

def track_duration(phase: str):
    """
    Decorator to track function duration.

    Usage:
        @track_duration("calc")
        def calculate_metrics(...):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with SCAN_DURATION.labels(phase=phase).time():
                return func(*args, **kwargs)
        return wrapper
    return decorator


def track_api_call(source: str):
    """
    Decorator to track API calls.

    Usage:
        @track_api_call("goldsky")
        async def fetch_from_goldsky(...):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.monotonic()
            try:
                result = await func(*args, **kwargs)
                API_REQUESTS.labels(source=source, status="success").inc()
                return result
            except Exception as e:
                API_REQUESTS.labels(source=source, status="error").inc()
                raise
            finally:
                API_LATENCY.labels(source=source).observe(time.monotonic() - start)
        return wrapper
    return decorator


def record_scan_result(
    wallets_scanned: int,
    wallets_failed: int,
    wallets_no_trades: int,
    metrics_calculated: int,
    metrics_saved: int,
    classifications: Dict[str, int],
    duration_seconds: float,
):
    """Record results from a scan operation"""
    WALLETS_SCANNED.labels(status="success").inc(wallets_scanned)
    WALLETS_SCANNED.labels(status="failed").inc(wallets_failed)
    WALLETS_SCANNED.labels(status="no_trades").inc(wallets_no_trades)

    METRICS_CALCULATED.inc(metrics_calculated)
    METRICS_SAVED.inc(metrics_saved)

    for classification, count in classifications.items():
        CLASSIFICATIONS.labels(classification=classification).inc(count)

    if duration_seconds > 0:
        THROUGHPUT.set(wallets_scanned / duration_seconds)


def get_metrics_summary() -> Dict[str, Any]:
    """
    Get summary of all metrics.

    Returns dict suitable for logging or API response.
    """
    if not PROMETHEUS_AVAILABLE:
        return {"prometheus_available": False}

    # This is a simplified version - full implementation would
    # parse the prometheus metrics
    return {
        "prometheus_available": True,
        "note": "Use /metrics endpoint for full Prometheus metrics",
    }


def start_metrics_server(port: int = 9090):
    """
    Start Prometheus metrics HTTP server.

    Args:
        port: Port to serve metrics on
    """
    if not PROMETHEUS_AVAILABLE:
        logger.warning("Cannot start metrics server: prometheus_client not installed")
        return

    def run_server():
        start_http_server(port)
        logger.info(f"Prometheus metrics server started on port {port}")

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()


class MetricsContext:
    """
    Context manager for tracking operation metrics.

    Usage:
        with MetricsContext("fetch") as ctx:
            data = await fetch_data()
            ctx.set_count(len(data))
    """

    def __init__(self, phase: str):
        self.phase = phase
        self.start_time: Optional[float] = None
        self.count = 0

    def __enter__(self):
        self.start_time = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.monotonic() - self.start_time
        SCAN_DURATION.labels(phase=self.phase).observe(duration)
        return False

    def set_count(self, count: int):
        """Set the count of items processed"""
        self.count = count
