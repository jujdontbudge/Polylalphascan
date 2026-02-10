"""
Ranking Engine - Step 5: Maintains Top 100 leaderboards per metric.

Total slots: 117 metrics × 100 wallets = 11,700 ranking entries.
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from src.database.repositories import RankingRepository, MetricRepository
from src.database.models import get_session, Metric, MetricRanking, Wallet
from config.settings import scanner_config

logger = logging.getLogger(__name__)


@dataclass
class RankingUpdateResult:
    """Results from ranking update"""
    metric_id: int
    wallet_id: int
    new_rank: Optional[int]
    previous_rank: Optional[int]
    qualified: bool
    reason: str = ""


@dataclass
class FullRecalcResult:
    """Results from full ranking recalculation"""
    metrics_processed: int = 0
    rankings_updated: int = 0
    duration_seconds: float = 0
    errors: List[str] = field(default_factory=list)


class RankingEngine:
    """
    Engine for maintaining Top 100 rankings per metric.

    Update strategies:
    1. Incremental: When a wallet's metric changes, check if it qualifies
    2. Full recalc: Complete ranking recalculation (daily at 2 AM)

    Uses buffer system: Maintain Top 150 internally, expose Top 100.
    """

    def __init__(self):
        self.top_n = scanner_config.TOP_N  # 100
        self.buffer_n = scanner_config.BUFFER_N  # 150

    def update_wallet_rankings(
        self,
        wallet_id: int,
        metrics: Dict[int, float],
    ) -> List[RankingUpdateResult]:
        """
        Update rankings for a single wallet across all its metrics.

        Args:
            wallet_id: Wallet database ID
            metrics: Dict of metric_id to value

        Returns:
            List of RankingUpdateResult for each metric
        """
        results = []
        session = get_session()

        try:
            with RankingRepository(session) as ranking_repo, \
                 MetricRepository(session) as metric_repo:

                for metric_id, value in metrics.items():
                    if value is None:
                        continue

                    # Get metric info
                    metric = metric_repo.get_by_id(metric_id)
                    if not metric:
                        continue

                    # Check if qualifies and update
                    result = self._update_single_ranking(
                        ranking_repo,
                        metric_id,
                        wallet_id,
                        value,
                        metric.higher_is_better,
                    )
                    results.append(result)

        finally:
            session.close()

        return results

    def _update_single_ranking(
        self,
        repo: RankingRepository,
        metric_id: int,
        wallet_id: int,
        value: float,
        higher_is_better: bool,
    ) -> RankingUpdateResult:
        """Update ranking for a single metric-wallet pair"""
        result = RankingUpdateResult(
            metric_id=metric_id,
            wallet_id=wallet_id,
            new_rank=None,
            previous_rank=None,
            qualified=False,
        )

        # Check previous rank
        existing = repo.get_wallet_ranking(metric_id, wallet_id)
        if existing:
            result.previous_rank = existing.rank

        # Try to upsert ranking
        ranking = repo.upsert_ranking(
            metric_id=metric_id,
            wallet_id=wallet_id,
            value=value,
            higher_is_better=higher_is_better,
        )

        if ranking:
            result.qualified = True
            result.new_rank = ranking.rank
            result.reason = "Qualified for Top 100"

            if result.previous_rank:
                if result.new_rank < result.previous_rank:
                    result.reason = f"Moved up from #{result.previous_rank} to #{result.new_rank}"
                elif result.new_rank > result.previous_rank:
                    result.reason = f"Moved down from #{result.previous_rank} to #{result.new_rank}"
                else:
                    result.reason = f"Maintained #{result.new_rank}"
        else:
            result.reason = "Did not qualify for Top 100"

        return result

    def get_leaderboard(
        self,
        metric_id: int,
        limit: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Get leaderboard for a specific metric.

        Args:
            metric_id: Metric ID
            limit: Max entries to return (default: top_n)

        Returns:
            List of ranking entries with wallet info
        """
        limit = limit or self.top_n
        session = get_session()

        try:
            with RankingRepository(session) as repo:
                rankings = repo.get_top_n_with_wallets(metric_id, n=limit)

            return [
                {
                    "rank": r.rank,
                    "wallet_address": w.address,
                    "wallet_id": w.id,
                    "value": r.value,
                    "percentile": r.percentile,
                    "classification": w.classification,
                    "quant_score": w.quant_score,
                    "updated_at": r.updated_at.isoformat() if r.updated_at else None,
                }
                for r, w in rankings
            ]
        finally:
            session.close()

    def get_wallet_leaderboard_positions(
        self,
        wallet_id: int,
    ) -> List[Dict[str, Any]]:
        """
        Get all leaderboard positions for a wallet.

        Args:
            wallet_id: Wallet database ID

        Returns:
            List of metric rankings the wallet appears in
        """
        session = get_session()

        try:
            with RankingRepository(session) as ranking_repo, \
                 MetricRepository(session) as metric_repo:

                rankings = ranking_repo.get_wallet_rankings(wallet_id)
                metrics_map = {m.id: m for m in metric_repo.get_all()}

            return [
                {
                    "metric_id": r.metric_id,
                    "metric_code": metrics_map[r.metric_id].code if r.metric_id in metrics_map else None,
                    "metric_name": metrics_map[r.metric_id].name if r.metric_id in metrics_map else None,
                    "rank": r.rank,
                    "value": r.value,
                    "percentile": r.percentile,
                }
                for r in rankings
            ]
        finally:
            session.close()

    def full_recalculation(self) -> FullRecalcResult:
        """
        Perform full ranking recalculation for all metrics.

        This should be run daily (e.g., at 2 AM) to catch edge cases
        and ensure ranking accuracy.

        Returns:
            FullRecalcResult with summary
        """
        result = FullRecalcResult()
        start_time = datetime.now()

        logger.info("Starting full ranking recalculation...")

        session = get_session()
        try:
            with RankingRepository(session) as ranking_repo:
                recalc_results = ranking_repo.rebuild_all_rankings()

            # recalc_results is {time_period: {metric_id: count}}
            all_metric_ids = set()
            total_rankings = 0
            for period_results in recalc_results.values():
                all_metric_ids.update(period_results.keys())
                total_rankings += sum(period_results.values())
            result.metrics_processed = len(all_metric_ids)
            result.rankings_updated = total_rankings

        except Exception as e:
            error_msg = f"Error during full recalculation: {str(e)}"
            logger.error(error_msg)
            result.errors.append(error_msg)
        finally:
            session.close()

        result.duration_seconds = (datetime.now() - start_time).total_seconds()

        logger.info(
            f"Full recalculation complete: {result.metrics_processed} metrics, "
            f"{result.rankings_updated} rankings in {result.duration_seconds:.1f}s"
        )

        return result

    def recalculate_metric(self, metric_id: int) -> int:
        """
        Recalculate rankings for a single metric.

        Args:
            metric_id: Metric ID to recalculate

        Returns:
            Number of rankings updated
        """
        session = get_session()
        try:
            with RankingRepository(session) as ranking_repo, \
                 MetricRepository(session) as metric_repo:

                metric = metric_repo.get_by_id(metric_id)
                if not metric:
                    logger.warning(f"Metric {metric_id} not found")
                    return 0

                count = ranking_repo.rebuild_rankings(
                    metric_id,
                    metric.higher_is_better,
                )
                return count
        finally:
            session.close()

    def update_percentiles(self, metric_id: int = None) -> int:
        """
        Update percentiles for rankings.

        Args:
            metric_id: Specific metric or None for all

        Returns:
            Number of rankings updated
        """
        session = get_session()
        total_updated = 0

        try:
            with MetricRepository(session) as metric_repo:
                if metric_id:
                    metrics = [metric_repo.get_by_id(metric_id)]
                else:
                    metrics = metric_repo.get_all()

                for metric in metrics:
                    if metric:
                        count = metric_repo.calculate_percentiles(metric.id)
                        total_updated += count

        finally:
            session.close()

        return total_updated

    def get_ranking_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all rankings"""
        session = get_session()
        try:
            with RankingRepository(session) as repo:
                return repo.get_ranking_summary()
        finally:
            session.close()

    def get_top_wallets_across_metrics(
        self,
        min_rankings: int = 5,
        max_avg_rank: float = 50,
    ) -> List[Dict[str, Any]]:
        """
        Find wallets that appear in Top 100 across multiple metrics.

        Args:
            min_rankings: Minimum number of Top 100 appearances
            max_avg_rank: Maximum average rank to include

        Returns:
            List of elite wallets with their ranking stats
        """
        session = get_session()

        try:
            from sqlalchemy import select, func
            from src.database.models import MetricRanking, Wallet

            # Query wallets with their ranking counts and average ranks
            stmt = (
                select(
                    Wallet.id,
                    Wallet.address,
                    Wallet.classification,
                    Wallet.quant_score,
                    func.count(MetricRanking.id).label("ranking_count"),
                    func.avg(MetricRanking.rank).label("avg_rank"),
                    func.min(MetricRanking.rank).label("best_rank"),
                )
                .join(MetricRanking, Wallet.id == MetricRanking.wallet_id)
                .group_by(Wallet.id)
                .having(func.count(MetricRanking.id) >= min_rankings)
                .having(func.avg(MetricRanking.rank) <= max_avg_rank)
                .order_by(func.count(MetricRanking.id).desc(), func.avg(MetricRanking.rank))
            )

            results = session.execute(stmt).all()

            return [
                {
                    "wallet_id": r.id,
                    "address": r.address,
                    "classification": r.classification,
                    "quant_score": r.quant_score,
                    "ranking_count": r.ranking_count,
                    "avg_rank": round(r.avg_rank, 1),
                    "best_rank": r.best_rank,
                }
                for r in results
            ]
        finally:
            session.close()

    def clear_all_rankings(self) -> int:
        """Clear all rankings (use with caution)"""
        session = get_session()
        try:
            with RankingRepository(session) as repo:
                return repo.clear_all_rankings()
        finally:
            session.close()
