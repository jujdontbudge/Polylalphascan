"""
Deep Scanner - Step 4: Full wallet analysis with 117 metrics.

Orchestrates data fetching and metric calculation for qualified wallets.
"""
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from src.api import DataClient
from src.database.repositories import WalletRepository, MetricRepository
from src.database.models import get_session, Wallet
from .metric_engine import MetricEngine, MetricResult

logger = logging.getLogger(__name__)


@dataclass
class DeepScanResult:
    """Results from deep scanning wallets"""
    wallets_scanned: int = 0
    wallets_failed: int = 0
    total_trades_processed: int = 0
    metrics_calculated: int = 0
    classifications: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    scan_duration_seconds: float = 0

    @property
    def success_rate(self) -> float:
        total = self.wallets_scanned + self.wallets_failed
        if total == 0:
            return 0.0
        return self.wallets_scanned / total


class DeepScanner:
    """
    Deep scanner for full wallet analysis.

    Fetches complete trade history and calculates all 117 metrics.
    """

    def __init__(
        self,
        data_client: DataClient = None,
        metric_engine: MetricEngine = None,
    ):
        self.data_client = data_client or DataClient()
        self.metric_engine = metric_engine or MetricEngine()

    def scan_wallet(
        self,
        wallet: Wallet,
        max_trades: int = None,
    ) -> Optional[MetricResult]:
        """
        Deep scan a single wallet.

        Args:
            wallet: Wallet database object
            max_trades: Maximum trades to fetch

        Returns:
            MetricResult or None if failed
        """
        try:
            # Fetch full activity
            activity = self.data_client.get_wallet_activity(
                wallet.address,
                include_positions=True,
                max_trades=max_trades,
            )

            if not activity.trades:
                logger.warning(f"No trades found for {wallet.address}")
                return None

            # Calculate metrics
            result = self.metric_engine.calculate_and_save(
                wallet_id=wallet.id,
                address=wallet.address,
                trades=activity.trades,
                positions=activity.positions,
            )

            return result

        except Exception as e:
            logger.error(f"Error scanning wallet {wallet.address}: {e}")
            return None

    def scan_wallets(
        self,
        wallets: List[Wallet],
        max_trades_per_wallet: int = None,
        progress_callback=None,
    ) -> DeepScanResult:
        """
        Deep scan multiple wallets.

        Args:
            wallets: List of Wallet objects
            max_trades_per_wallet: Max trades per wallet
            progress_callback: Optional callback(current, total, wallet_address)

        Returns:
            DeepScanResult with summary
        """
        result = DeepScanResult()
        result.classifications = {"bot": 0, "quant": 0, "human": 0, "unknown": 0}
        start_time = datetime.now()

        logger.info(f"Starting deep scan of {len(wallets)} wallets...")

        for i, wallet in enumerate(wallets):
            if progress_callback:
                progress_callback(i + 1, len(wallets), wallet.address)

            scan_result = self.scan_wallet(wallet, max_trades=max_trades_per_wallet)

            if scan_result:
                result.wallets_scanned += 1
                result.metrics_calculated += scan_result.total_calculated
                result.classifications[scan_result.classification] = \
                    result.classifications.get(scan_result.classification, 0) + 1

                if scan_result.errors:
                    result.errors.extend(scan_result.errors)
            else:
                result.wallets_failed += 1

        result.scan_duration_seconds = (datetime.now() - start_time).total_seconds()

        logger.info(
            f"Deep scan complete: {result.wallets_scanned} scanned, "
            f"{result.wallets_failed} failed, {result.metrics_calculated} metrics calculated"
        )

        return result

    def scan_all_wallets(
        self,
        batch_size: int = 100,
        limit: Optional[int] = None,
        max_trades_per_wallet: int = None,
        progress_callback=None,
    ) -> DeepScanResult:
        """
        Scan all wallets in the database in batches.

        Args:
            batch_size: Wallets per batch
            limit: Optional max wallets to scan (None = all)
            max_trades_per_wallet: Max trades to fetch per wallet (None = full history)
            progress_callback: Optional callback(scanned, total, wallet_address)

        Returns:
            DeepScanResult with aggregated summary
        """
        total_result = DeepScanResult()
        total_result.classifications = {"bot": 0, "quant": 0, "human": 0, "unknown": 0}
        start_time = datetime.now()

        session = get_session()
        try:
            with WalletRepository(session) as repo:
                offset = 0
                scanned = 0

                while True:
                    wallets = repo.get_all(limit=batch_size, offset=offset)
                    if not wallets:
                        break

                    if limit is not None:
                        remaining = max(0, limit - scanned)
                        if remaining == 0:
                            break
                        wallets = wallets[:remaining]

                    batch_result = self.scan_wallets(
                        wallets,
                        max_trades_per_wallet=max_trades_per_wallet,
                        progress_callback=progress_callback,
                    )

                    total_result.wallets_scanned += batch_result.wallets_scanned
                    total_result.wallets_failed += batch_result.wallets_failed
                    total_result.metrics_calculated += batch_result.metrics_calculated
                    total_result.errors.extend(batch_result.errors)

                    for cls, count in (batch_result.classifications or {}).items():
                        total_result.classifications[cls] = total_result.classifications.get(cls, 0) + count

                    scanned += len(wallets)
                    offset += len(wallets)

                    if limit is not None and scanned >= limit:
                        break

        finally:
            session.close()

        total_result.scan_duration_seconds = (datetime.now() - start_time).total_seconds()
        return total_result

    def scan_unscanned_wallets(
        self,
        limit: int = 100,
        max_trades_per_wallet: int = None,
    ) -> DeepScanResult:
        """
        Scan wallets that haven't been deep scanned yet.

        Args:
            limit: Maximum wallets to scan
            max_trades_per_wallet: Max trades per wallet

        Returns:
            DeepScanResult with summary
        """
        session = get_session()
        try:
            with WalletRepository(session) as repo:
                wallets = repo.get_unscanned(limit=limit)
        finally:
            session.close()

        return self.scan_wallets(wallets, max_trades_per_wallet)

    def scan_by_classification(
        self,
        classification: str,
        limit: int = 100,
        max_trades_per_wallet: int = None,
    ) -> DeepScanResult:
        """
        Rescan wallets of a specific classification.

        Args:
            classification: bot, quant, human, or unknown
            limit: Maximum wallets to scan
            max_trades_per_wallet: Max trades per wallet

        Returns:
            DeepScanResult with summary
        """
        session = get_session()
        try:
            with WalletRepository(session) as repo:
                wallets = repo.get_by_classification(classification, limit=limit)
        finally:
            session.close()

        return self.scan_wallets(wallets, max_trades_per_wallet)

    def rescan_elite_wallets(
        self,
        limit: int = 100,
        max_trades_per_wallet: int = None,
    ) -> DeepScanResult:
        """
        Rescan elite flagged wallets with latest data.

        Args:
            limit: Maximum wallets to scan
            max_trades_per_wallet: Max trades per wallet

        Returns:
            DeepScanResult with summary
        """
        session = get_session()
        try:
            with WalletRepository(session) as repo:
                wallets = repo.get_elite_wallets(limit=limit)
        finally:
            session.close()

        return self.scan_wallets(wallets, max_trades_per_wallet)

    def close(self):
        """Close resources"""
        self.data_client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
