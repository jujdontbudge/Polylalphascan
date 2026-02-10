"""Analyzer module for metrics calculation"""
from .metric_engine import MetricEngine
from .deep_scanner import DeepScanner
from .parallel_deep_scanner import (
    ParallelDeepScanner,
    ParallelScanResult,
    UltraScanner,
    scan_1000_wallets,
)

# Ultra-fast 117 metrics calculator (Numba-powered)
from .numba_metrics import (
    FastWalletData,
    calculate_117_metrics,
    calculate_117_metrics_from_trades,
    calculate_batch_117,
    calculate_quant_score,
    METRIC_NAMES,
)

# Ultra-fast scanner with 117 metrics support
from .ultra_fast_scanner import (
    UltraFastScanner,
    UltraFastResult,
    ultra_fast_scan,
    ultra_fast_scan_117,
)

# Fast 30-metric calculator (for backwards compatibility)
from .fast_metrics import (
    FastWalletData as FastWalletData30,
    calculate_fast_metrics,
    FAST_METRIC_IDS,
)

# Try to import optimized scanner (requires additional dependencies)
try:
    from .optimized_scanner import (
        OptimizedScanner,
        OptimizedScanResult,
        UltraOptimizedScanner,
        scan_1000_wallets_optimized,
    )
    OPTIMIZED_SCANNER_AVAILABLE = True
except ImportError:
    OPTIMIZED_SCANNER_AVAILABLE = False

__all__ = [
    "MetricEngine",
    "DeepScanner",
    # High-throughput scanners
    "ParallelDeepScanner",
    "ParallelScanResult",
    "UltraScanner",
    "scan_1000_wallets",
    # Ultra-fast 117 metrics (Numba-powered, 50,000+ wallets/sec)
    "FastWalletData",
    "calculate_117_metrics",
    "calculate_117_metrics_from_trades",
    "calculate_batch_117",
    "calculate_quant_score",
    "METRIC_NAMES",
    # Ultra-fast scanner
    "UltraFastScanner",
    "UltraFastResult",
    "ultra_fast_scan",
    "ultra_fast_scan_117",
    # Fast 30 metrics (backwards compatibility)
    "FastWalletData30",
    "calculate_fast_metrics",
    "FAST_METRIC_IDS",
    # Availability flags
    "OPTIMIZED_SCANNER_AVAILABLE",
]

# Conditionally add optimized scanner exports
if OPTIMIZED_SCANNER_AVAILABLE:
    __all__.extend([
        "OptimizedScanner",
        "OptimizedScanResult",
        "UltraOptimizedScanner",
        "scan_1000_wallets_optimized",
    ])
