"""
Celery Scanner Worker - Distributed wallet scanning for horizontal scaling.

Enables processing 1000+ wallets/second across multiple machines.

Usage:
    # Start worker
    celery -A src.workers.scanner_worker worker --loglevel=info

    # Send tasks
    from src.workers.scanner_worker import scan_wallet_batch
    result = scan_wallet_batch.delay(["0x123...", "0x456..."])
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import celery
try:
    from celery import Celery
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    logger.warning("celery not installed. Install with: pip install celery")

    # Create dummy decorator
    class Celery:
        def __init__(self, *args, **kwargs):
            pass

        def task(self, *args, **kwargs):
            def decorator(func):
                func.delay = lambda *a, **k: func(*a, **k)
                func.apply_async = lambda *a, **k: func(*a, **k)
                return func
            return decorator


# Celery configuration
app = Celery(
    "polyscan_workers",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,  # 10 minutes max
    worker_prefetch_multiplier=1,
    worker_concurrency=4,
)


@app.task(bind=True, name="scan_wallet_batch")
def scan_wallet_batch(
    self,
    addresses: List[str],
    max_trades_per_wallet: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Scan a batch of wallet addresses.

    This task can be distributed across multiple Celery workers.

    Args:
        addresses: List of wallet addresses to scan
        max_trades_per_wallet: Maximum trades per wallet

    Returns:
        Dict with scan results and statistics
    """
    import time
    start_time = time.monotonic()

    try:
        from src.core.analyzer.parallel_deep_scanner import ParallelDeepScanner
        from src.database.repositories import WalletRepository
        from src.database.models import get_session

        # Get or create wallets in database
        session = get_session()
        try:
            with WalletRepository(session) as repo:
                wallets = []
                for address in addresses:
                    wallet = repo.get_or_create(address)
                    wallets.append(wallet)
        finally:
            session.close()

        # Run scanner
        scanner = ParallelDeepScanner()
        result = scanner.scan_wallets(
            wallets,
            max_trades_per_wallet=max_trades_per_wallet,
        )

        return {
            "success": True,
            "wallets_scanned": result.wallets_scanned,
            "wallets_failed": result.wallets_failed,
            "metrics_calculated": result.metrics_calculated,
            "metrics_saved": result.metrics_saved,
            "classifications": result.classifications,
            "duration_seconds": time.monotonic() - start_time,
            "wallets_per_second": result.wallets_per_second,
        }

    except Exception as e:
        logger.exception(f"Scan batch failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "duration_seconds": time.monotonic() - start_time,
        }


@app.task(bind=True, name="scan_unscanned_batch")
def scan_unscanned_batch(
    self,
    limit: int = 100,
    max_trades_per_wallet: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Scan a batch of unscanned wallets from the database.

    Args:
        limit: Maximum wallets to scan
        max_trades_per_wallet: Maximum trades per wallet

    Returns:
        Dict with scan results
    """
    import time
    start_time = time.monotonic()

    try:
        from src.core.analyzer.parallel_deep_scanner import ParallelDeepScanner
        from src.database.repositories import WalletRepository
        from src.database.models import get_session

        # Get unscanned wallets
        session = get_session()
        try:
            with WalletRepository(session) as repo:
                wallets = repo.get_unscanned(limit=limit)
        finally:
            session.close()

        if not wallets:
            return {
                "success": True,
                "wallets_scanned": 0,
                "message": "No unscanned wallets found",
                "duration_seconds": time.monotonic() - start_time,
            }

        # Run scanner
        scanner = ParallelDeepScanner()
        result = scanner.scan_wallets(
            wallets,
            max_trades_per_wallet=max_trades_per_wallet,
        )

        return {
            "success": True,
            "wallets_scanned": result.wallets_scanned,
            "wallets_failed": result.wallets_failed,
            "metrics_calculated": result.metrics_calculated,
            "metrics_saved": result.metrics_saved,
            "classifications": result.classifications,
            "duration_seconds": time.monotonic() - start_time,
            "wallets_per_second": result.wallets_per_second,
        }

    except Exception as e:
        logger.exception(f"Scan unscanned batch failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "duration_seconds": time.monotonic() - start_time,
        }


@app.task(bind=True, name="discover_and_scan")
def discover_and_scan(
    self,
    min_volume: float = 500.0,
    max_markets: int = 100,
    max_wallets: int = 1000,
) -> Dict[str, Any]:
    """
    Discover wallets from markets and scan them.

    Full pipeline task for harvesting + scanning.

    Args:
        min_volume: Minimum market volume filter
        max_markets: Maximum markets to scan
        max_wallets: Maximum wallets to scan

    Returns:
        Dict with harvest and scan results
    """
    import asyncio
    import time
    start_time = time.monotonic()

    try:
        from src.core.analyzer.parallel_deep_scanner import UltraScanner

        scanner = UltraScanner()
        harvest_result, scan_result = asyncio.run(
            scanner.discover_and_scan(
                min_volume=min_volume,
                max_markets=max_markets,
                max_wallets=max_wallets,
            )
        )

        return {
            "success": True,
            "wallets_discovered": len(harvest_result.wallets),
            "wallets_scanned": scan_result.wallets_scanned,
            "wallets_failed": scan_result.wallets_failed,
            "metrics_calculated": scan_result.metrics_calculated,
            "classifications": scan_result.classifications,
            "duration_seconds": time.monotonic() - start_time,
            "wallets_per_second": scan_result.wallets_per_second,
        }

    except Exception as e:
        logger.exception(f"Discover and scan failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "duration_seconds": time.monotonic() - start_time,
        }


def distribute_scan(
    addresses: List[str],
    batch_size: int = 100,
    max_trades_per_wallet: Optional[int] = None,
) -> List[Any]:
    """
    Distribute scanning across multiple workers.

    Splits addresses into batches and dispatches to Celery workers.

    Args:
        addresses: All wallet addresses to scan
        batch_size: Addresses per batch/task
        max_trades_per_wallet: Maximum trades per wallet

    Returns:
        List of AsyncResult objects
    """
    results = []

    for i in range(0, len(addresses), batch_size):
        batch = addresses[i:i + batch_size]
        result = scan_wallet_batch.delay(batch, max_trades_per_wallet)
        results.append(result)

    return results


def wait_for_results(results: List[Any], timeout: int = 600) -> List[Dict[str, Any]]:
    """
    Wait for all distributed tasks to complete.

    Args:
        results: List of AsyncResult objects
        timeout: Maximum wait time in seconds

    Returns:
        List of result dicts
    """
    outputs = []

    for result in results:
        try:
            output = result.get(timeout=timeout)
            outputs.append(output)
        except Exception as e:
            outputs.append({
                "success": False,
                "error": str(e),
            })

    return outputs
