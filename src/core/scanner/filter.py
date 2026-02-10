"""Step 3: Lite Filter - Quick filtering before deep scan"""
import logging
from typing import List, Set, Dict, Any, Tuple
from dataclasses import dataclass, field

from src.api import DataClient
from src.database.repositories import WalletRepository
from src.database.models import get_session
from config.settings import scanner_config

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Results from lite filtering"""
    qualified: List[str] = field(default_factory=list)  # Wallet addresses
    disqualified: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def qualified_count(self) -> int:
        return len(self.qualified)

    @property
    def disqualified_count(self) -> int:
        return len(self.disqualified)

    @property
    def pass_rate(self) -> float:
        total = self.qualified_count + self.disqualified_count
        if total == 0:
            return 0.0
        return self.qualified_count / total


@dataclass
class WalletStats:
    """Quick stats for a wallet"""
    address: str
    volume: float
    trade_count: int
    qualified: bool
    reason: str = ""


class LiteFilter:
    """
    Step 3: Quick filtering of wallets before deep scan.

    Applies minimum thresholds to avoid wasting resources on inactive wallets:
    - Volume < $500 → DISCARD
    - Trade Count < 10 → DISCARD
    """

    def __init__(
        self,
        data_client: DataClient = None,
        min_volume: float = None,
        min_trades: int = None,
    ):
        self.data_client = data_client or DataClient()
        self.min_volume = min_volume or scanner_config.MIN_VOLUME_USD
        self.min_trades = min_trades or scanner_config.MIN_TRADE_COUNT

    def check_wallet(self, address: str) -> WalletStats:
        """
        Check if a single wallet qualifies for deep scan.

        Args:
            address: Wallet address

        Returns:
            WalletStats with qualification status
        """
        try:
            stats = self.data_client.get_wallet_stats(address)

            volume = stats.get("volume", 0)
            trade_count = stats.get("trade_count", 0)

            qualified = True
            reason = ""

            if volume < self.min_volume:
                qualified = False
                reason = f"Volume ${volume:.2f} < ${self.min_volume}"
            elif trade_count < self.min_trades:
                qualified = False
                reason = f"Trades {trade_count} < {self.min_trades}"

            return WalletStats(
                address=address,
                volume=volume,
                trade_count=trade_count,
                qualified=qualified,
                reason=reason,
            )

        except Exception as e:
            logger.error(f"Error checking wallet {address}: {e}")
            return WalletStats(
                address=address,
                volume=0,
                trade_count=0,
                qualified=False,
                reason=f"Error: {str(e)}",
            )

    def filter_wallets(
        self,
        addresses: List[str],
        save_to_db: bool = True,
    ) -> FilterResult:
        """
        Filter a list of wallet addresses.

        Args:
            addresses: Wallet addresses to filter
            save_to_db: Whether to save qualified wallets to database

        Returns:
            FilterResult with qualified and disqualified lists
        """
        result = FilterResult()
        result.stats["total_input"] = len(addresses)
        result.stats["min_volume"] = self.min_volume
        result.stats["min_trades"] = self.min_trades

        logger.info(f"Filtering {len(addresses)} wallets...")

        volume_fails = 0
        trade_fails = 0
        error_fails = 0

        for i, address in enumerate(addresses):
            if (i + 1) % 50 == 0:
                logger.info(f"Filtering wallet {i + 1}/{len(addresses)}...")

            stats = self.check_wallet(address)

            if stats.qualified:
                result.qualified.append(address)
            else:
                result.disqualified.append(address)

                # Track failure reasons
                if "Volume" in stats.reason:
                    volume_fails += 1
                elif "Trades" in stats.reason:
                    trade_fails += 1
                else:
                    error_fails += 1

        # Update stats
        result.stats["qualified"] = result.qualified_count
        result.stats["disqualified"] = result.disqualified_count
        result.stats["pass_rate"] = result.pass_rate
        result.stats["volume_fails"] = volume_fails
        result.stats["trade_fails"] = trade_fails
        result.stats["error_fails"] = error_fails

        logger.info(
            f"Filter complete: {result.qualified_count} qualified, "
            f"{result.disqualified_count} disqualified "
            f"(pass rate: {result.pass_rate:.1%})"
        )

        # Save to database
        if save_to_db and result.qualified:
            self._save_qualified_wallets(result.qualified)

        return result

    def _save_qualified_wallets(self, addresses: List[str]):
        """Save qualified wallets to database"""
        try:
            session = get_session()
            with WalletRepository(session) as repo:
                created = repo.bulk_create(addresses)
                logger.info(f"Saved {len(created)} new wallets to database")
        except Exception as e:
            logger.error(f"Error saving wallets to database: {e}")

    def filter_and_return_stats(
        self,
        addresses: List[str],
    ) -> Tuple[List[str], List[WalletStats]]:
        """
        Filter wallets and return detailed stats.

        Args:
            addresses: Wallet addresses to filter

        Returns:
            Tuple of (qualified addresses, all wallet stats)
        """
        qualified = []
        all_stats = []

        for address in addresses:
            stats = self.check_wallet(address)
            all_stats.append(stats)
            if stats.qualified:
                qualified.append(address)

        return qualified, all_stats

    def quick_filter(
        self,
        addresses: List[str],
        sample_size: int = 100,
    ) -> Tuple[List[str], float]:
        """
        Quick filter with sampling to estimate pass rate.

        Args:
            addresses: Wallet addresses
            sample_size: Sample size for estimation

        Returns:
            Tuple of (estimated qualified, estimated pass rate)
        """
        import random

        if len(addresses) <= sample_size:
            result = self.filter_wallets(addresses, save_to_db=False)
            return result.qualified, result.pass_rate

        # Sample
        sample = random.sample(addresses, sample_size)
        result = self.filter_wallets(sample, save_to_db=False)

        # Estimate
        estimated_qualified = int(len(addresses) * result.pass_rate)

        return result.qualified, result.pass_rate

    def get_thresholds(self) -> Dict[str, Any]:
        """Get current filter thresholds"""
        return {
            "min_volume_usd": self.min_volume,
            "min_trade_count": self.min_trades,
        }

    def set_thresholds(
        self,
        min_volume: float = None,
        min_trades: int = None,
    ):
        """Update filter thresholds"""
        if min_volume is not None:
            self.min_volume = min_volume
        if min_trades is not None:
            self.min_trades = min_trades

    def close(self):
        """Close API client"""
        self.data_client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
