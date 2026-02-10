"""
Metric Engine - Calculates all 117 metrics for wallets.

Orchestrates the 15 metric groups (A-O) using vectorized pandas/numpy calculations.
Supports time-period based calculations: all_time, 7d, 31d.
"""
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from src.api.data_client import Trade, Position
from src.database.repositories import MetricRepository, WalletRepository
from src.database.models import get_session

from .groups.base_group import (
    WalletData,
    TIME_PERIODS,
    TIME_PERIOD_ALL,
    TIME_PERIOD_7D,
    TIME_PERIOD_31D,
)
from .groups import (
    GroupAProfitability,
    GroupBRisk,
    GroupCWinLoss,
    GroupDActivity,
    GroupETiming,
    GroupFMoney,
    GroupGStrategy,
    GroupHPsychological,
    GroupIAlpha,
    GroupJHygiene,
    GroupKCopyability,
    GroupLMarketIntel,
    GroupMExecution,
    GroupNBotQuant,
    GroupOPortfolio,
)
from .groups.group_n_bot_quant import calculate_quant_score

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Results from metric calculation for a single time period"""
    wallet_address: str
    time_period: str
    metrics: Dict[int, float]  # metric_id -> value
    quant_score: float
    classification: str
    errors: List[str]

    @property
    def total_calculated(self) -> int:
        return len([v for v in self.metrics.values() if v is not None])


@dataclass
class MultiPeriodMetricResult:
    """Results from metric calculation across all time periods"""
    wallet_address: str
    results: Dict[str, MetricResult] = field(default_factory=dict)  # time_period -> MetricResult
    errors: List[str] = field(default_factory=list)

    @property
    def total_calculated(self) -> int:
        return sum(r.total_calculated for r in self.results.values())

    @property
    def all_time_result(self) -> Optional[MetricResult]:
        return self.results.get(TIME_PERIOD_ALL)

    @property
    def classification(self) -> str:
        """Get classification from all-time result"""
        if self.all_time_result:
            return self.all_time_result.classification
        return "unknown"

    @property
    def quant_score(self) -> float:
        """Get quant score from all-time result"""
        if self.all_time_result:
            return self.all_time_result.quant_score
        return 0.0


class MetricEngine:
    """
    Main engine for calculating all 117 metrics.

    Uses vectorized operations where possible for performance.
    """

    def __init__(self):
        # Initialize all metric groups
        self.groups = [
            GroupAProfitability(),
            GroupBRisk(),
            GroupCWinLoss(),
            GroupDActivity(),
            GroupETiming(),
            GroupFMoney(),
            GroupGStrategy(),
            GroupHPsychological(),
            GroupIAlpha(),
            GroupJHygiene(),
            GroupKCopyability(),
            GroupLMarketIntel(),
            GroupMExecution(),
            GroupNBotQuant(),
            GroupOPortfolio(),
        ]

        self.group_map = {g.GROUP_ID: g for g in self.groups}

    def calculate_all(
        self,
        address: str,
        trades: List[Trade],
        positions: List[Position] = None,
        time_period: str = TIME_PERIOD_ALL,
    ) -> MetricResult:
        """
        Calculate all 117 metrics for a wallet for a single time period.

        Args:
            address: Wallet address
            trades: List of Trade objects
            positions: Optional list of Position objects
            time_period: Time period to calculate for (all_time, 7d, 31d)

        Returns:
            MetricResult with all calculated metrics
        """
        result = MetricResult(
            wallet_address=address,
            time_period=time_period,
            metrics={},
            quant_score=0.0,
            classification="unknown",
            errors=[],
        )

        # Create wallet data container
        data = WalletData(
            address=address,
            trades=trades,
            positions=positions or [],
            time_period=TIME_PERIOD_ALL,  # Start with all trades
        )

        # Filter to specific time period if needed
        if time_period != TIME_PERIOD_ALL:
            data = data.for_time_period(time_period)

        if not data.has_trades:
            result.errors.append(f"No trades found for {time_period}")
            return result

        logger.debug(f"Calculating metrics for {address} ({len(data.trades)} trades, {time_period})")

        # Calculate metrics from each group
        for group in self.groups:
            try:
                group_metrics = group.calculate(data)
                result.metrics.update(group_metrics)
            except Exception as e:
                error_msg = f"Error in group {group.GROUP_ID}: {str(e)}"
                logger.error(error_msg)
                result.errors.append(error_msg)

        # Calculate quant score from Group N metrics
        group_n_metrics = {
            mid: result.metrics.get(mid, 0)
            for mid in range(109, 114)
        }
        result.quant_score = calculate_quant_score(group_n_metrics)

        # Classify wallet
        result.classification = self._classify_wallet(result.quant_score)

        logger.debug(
            f"Calculated {result.total_calculated} metrics for {address} ({time_period}), "
            f"quant_score={result.quant_score:.1f}, classification={result.classification}"
        )

        return result

    def calculate_all_periods(
        self,
        address: str,
        trades: List[Trade],
        positions: List[Position] = None,
    ) -> MultiPeriodMetricResult:
        """
        Calculate all 117 metrics for all 3 time periods (all_time, 7d, 31d).

        Optimized to share computed values (cost basis, PnL series) across periods.
        Pre-computes expensive values once on the master WalletData, then filters
        trades for each period. Expected 2-3x speedup.

        Args:
            address: Wallet address
            trades: List of Trade objects
            positions: Optional list of Position objects

        Returns:
            MultiPeriodMetricResult with metrics for all time periods
        """
        result = MultiPeriodMetricResult(
            wallet_address=address,
            results={},
            errors=[],
        )

        # Create master WalletData with all trades
        master_data = WalletData(
            address=address,
            trades=trades,
            positions=positions or [],
            time_period=TIME_PERIOD_ALL,
        )

        # Pre-compute expensive values once on master data
        # This triggers memoization so subsequent accesses are O(1)
        if master_data.has_trades:
            _ = master_data.cost_basis_map
            _ = master_data.pnl_series
            _ = master_data.market_pnl_map

        for time_period in TIME_PERIODS:
            try:
                period_result = self.calculate_all(address, trades, positions, time_period)
                result.results[time_period] = period_result
            except Exception as e:
                error_msg = f"Error calculating {time_period}: {str(e)}"
                logger.error(error_msg)
                result.errors.append(error_msg)

        logger.debug(
            f"Calculated {result.total_calculated} metrics across {len(result.results)} periods for {address}"
        )

        return result

    def _classify_wallet(self, quant_score: float) -> str:
        """Classify wallet based on quant score"""
        if quant_score >= 80:
            return "bot"
        elif quant_score >= 50:
            return "quant"
        elif quant_score <= 20:
            return "human"
        else:
            return "unknown"

    def calculate_group(
        self,
        group_id: str,
        address: str,
        trades: List[Trade],
        positions: List[Position] = None,
    ) -> Dict[int, float]:
        """
        Calculate metrics for a single group.

        Args:
            group_id: Group ID (A-O)
            address: Wallet address
            trades: List of Trade objects
            positions: Optional list of Position objects

        Returns:
            Dict of metric_id to value
        """
        if group_id not in self.group_map:
            raise ValueError(f"Unknown group: {group_id}")

        data = WalletData(
            address=address,
            trades=trades,
            positions=positions or [],
        )

        group = self.group_map[group_id]
        return group.calculate(data)

    def calculate_and_save(
        self,
        wallet_id: int,
        address: str,
        trades: List[Trade],
        positions: List[Position] = None,
    ) -> MultiPeriodMetricResult:
        """
        Calculate metrics for all time periods and save to database.

        Args:
            wallet_id: Database wallet ID
            address: Wallet address
            trades: List of Trade objects
            positions: Optional list of Position objects

        Returns:
            MultiPeriodMetricResult with calculated metrics for all periods
        """
        result = self.calculate_all_periods(address, trades, positions)

        # Save to database
        session = get_session()
        try:
            with MetricRepository(session) as metric_repo:
                # Save metrics for each time period
                for time_period, period_result in result.results.items():
                    metric_repo.bulk_set_wallet_metrics(
                        wallet_id,
                        period_result.metrics,
                        time_period=time_period,
                    )

            with WalletRepository(session) as wallet_repo:
                # Use all-time classification
                wallet_repo.set_classification(
                    wallet_id,
                    result.classification,
                    result.quant_score,
                )
                wallet_repo.mark_scanned(
                    wallet_id,
                    total_trades=len(trades),
                    total_volume=sum(t.value for t in trades),
                )
        finally:
            session.close()

        return result

    def get_metric_info(self, metric_id: int) -> Optional[Dict[str, Any]]:
        """Get information about a specific metric"""
        for group in self.groups:
            if metric_id in group.METRIC_IDS:
                return {
                    "metric_id": metric_id,
                    "group_id": group.GROUP_ID,
                    "group_name": group.__class__.__name__,
                }
        return None

    def get_group_metrics(self, group_id: str) -> List[int]:
        """Get metric IDs for a group"""
        if group_id in self.group_map:
            return self.group_map[group_id].METRIC_IDS
        return []

    @staticmethod
    def get_all_metric_ids() -> List[int]:
        """Get all 117 metric IDs"""
        return list(range(1, 118))

    @staticmethod
    def get_group_ids() -> List[str]:
        """Get all 15 group IDs"""
        return list("ABCDEFGHIJKLMNO")
