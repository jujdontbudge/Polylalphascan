"""Step 2: Wallet Harvesting - Extract wallet addresses from market activity"""
import logging
from typing import List, Set, Dict, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.api import DataClient, CLOBClient
from config.settings import scanner_config

logger = logging.getLogger(__name__)


@dataclass
class HarvestResult:
    """Results from wallet harvesting"""
    wallets: Set[str] = field(default_factory=set)
    conditions_scanned: int = 0
    trades_processed: int = 0
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_wallets(self) -> int:
        return len(self.wallets)

    def merge(self, other: "HarvestResult"):
        """Merge another result into this one"""
        self.wallets.update(other.wallets)
        self.conditions_scanned += other.conditions_scanned
        self.trades_processed += other.trades_processed


class WalletHarvester:
    """
    Step 2: Extract wallet addresses from market trades and order books.

    Scans trade history for discovered markets to find active traders.
    """

    def __init__(
        self,
        data_client: DataClient = None,
        clob_client: CLOBClient = None,
    ):
        self.data_client = data_client or DataClient()
        self.clob_client = clob_client or CLOBClient()

    def harvest_from_condition(
        self,
        condition_id: str,
        max_trades: int = 1000,
    ) -> HarvestResult:
        """
        Harvest wallets from a single market condition.

        Args:
            condition_id: Market condition ID
            max_trades: Maximum trades to scan

        Returns:
            HarvestResult with discovered wallets
        """
        result = HarvestResult()

        try:
            wallets = self.data_client.extract_wallets_from_trades(
                condition_id=condition_id,
                max_trades=max_trades,
            )
            result.wallets.update(wallets)
            result.conditions_scanned = 1
            result.trades_processed = len(wallets)  # Approximate

        except Exception as e:
            logger.error(f"Error harvesting from condition {condition_id}: {e}")

        return result

    def harvest_from_conditions(
        self,
        condition_ids: List[str],
        max_trades_per_condition: int = 500,
        max_workers: int = 5,
    ) -> HarvestResult:
        """
        Harvest wallets from multiple market conditions.

        Args:
            condition_ids: List of condition IDs
            max_trades_per_condition: Max trades per condition
            max_workers: Parallel workers for harvesting

        Returns:
            HarvestResult with all discovered wallets
        """
        result = HarvestResult()
        result.stats["conditions_requested"] = len(condition_ids)

        logger.info(f"Harvesting wallets from {len(condition_ids)} conditions...")

        # Process conditions (sequential to respect rate limits)
        for i, condition_id in enumerate(condition_ids):
            if (i + 1) % 10 == 0:
                logger.info(f"Processing condition {i + 1}/{len(condition_ids)}...")

            condition_result = self.harvest_from_condition(
                condition_id,
                max_trades=max_trades_per_condition,
            )
            result.merge(condition_result)

        result.stats["wallets_discovered"] = result.total_wallets
        result.stats["conditions_scanned"] = result.conditions_scanned

        logger.info(
            f"Harvest complete: {result.total_wallets} unique wallets "
            f"from {result.conditions_scanned} conditions"
        )

        return result

    def harvest_recent_traders(
        self,
        limit: int = 1000,
        market_ids: List[str] = None,
    ) -> HarvestResult:
        """
        Harvest wallets from recent trades.

        Args:
            limit: Max trades to scan
            market_ids: Optional market filter

        Returns:
            HarvestResult with discovered wallets
        """
        result = HarvestResult()

        try:
            trades = self.data_client.get_recent_trades(
                limit=limit,
                market_ids=market_ids,
            )

            for trade in trades:
                if trade.wallet_address:
                    result.wallets.add(trade.wallet_address.lower())

            result.trades_processed = len(trades)

        except Exception as e:
            logger.error(f"Error harvesting recent traders: {e}")

        logger.info(f"Harvested {result.total_wallets} wallets from recent trades")
        return result

    def get_unique_wallets(
        self,
        condition_ids: List[str],
        existing_wallets: Set[str] = None,
        max_trades_per_condition: int = 500,
    ) -> Set[str]:
        """
        Get unique wallets not already in existing set.

        Args:
            condition_ids: Conditions to scan
            existing_wallets: Already known wallets to exclude
            max_trades_per_condition: Max trades per condition

        Returns:
            Set of new wallet addresses
        """
        existing = existing_wallets or set()
        result = self.harvest_from_conditions(
            condition_ids,
            max_trades_per_condition=max_trades_per_condition,
        )

        new_wallets = result.wallets - existing
        logger.info(f"Found {len(new_wallets)} new wallets")
        return new_wallets

    def estimate_wallet_count(
        self,
        condition_ids: List[str],
        sample_size: int = 10,
        trades_per_sample: int = 100,
    ) -> int:
        """
        Estimate total unique wallets by sampling.

        Args:
            condition_ids: Conditions to sample from
            sample_size: Number of conditions to sample
            trades_per_sample: Trades per sample condition

        Returns:
            Estimated total unique wallets
        """
        import random

        sample_ids = random.sample(
            condition_ids,
            min(sample_size, len(condition_ids)),
        )

        result = self.harvest_from_conditions(
            sample_ids,
            max_trades_per_condition=trades_per_sample,
        )

        # Extrapolate
        wallets_per_condition = result.total_wallets / max(1, result.conditions_scanned)
        estimated_total = int(wallets_per_condition * len(condition_ids) * 0.7)  # Overlap factor

        return estimated_total

    def close(self):
        """Close API clients"""
        self.data_client.close()
        self.clob_client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
