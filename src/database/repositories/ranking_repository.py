"""Repository for ranking data access"""
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy import select, delete, func, and_
from sqlalchemy.orm import Session, joinedload

from src.database.models import Metric, MetricRanking, Wallet, WalletMetric, get_session, TIME_PERIOD_ALL, TIME_PERIODS
from config.settings import scanner_config


class RankingRepository:
    """Data access layer for Top 100 rankings"""

    def __init__(self, session: Optional[Session] = None):
        self.session = session or get_session()
        self._owns_session = session is None
        self.top_n = scanner_config.TOP_N  # 100
        self.buffer_n = scanner_config.BUFFER_N  # 150

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._owns_session:
            self.session.close()

    def get_ranking(
        self,
        metric_id: int,
        rank: int,
        time_period: str = TIME_PERIOD_ALL,
    ) -> Optional[MetricRanking]:
        """Get specific ranking entry for a time period"""
        stmt = select(MetricRanking).where(
            MetricRanking.metric_id == metric_id,
            MetricRanking.rank == rank,
            MetricRanking.time_period == time_period,
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def get_wallet_ranking(
        self,
        metric_id: int,
        wallet_id: int,
        time_period: str = TIME_PERIOD_ALL,
    ) -> Optional[MetricRanking]:
        """Get ranking for a specific wallet in a metric for a time period"""
        stmt = select(MetricRanking).where(
            MetricRanking.metric_id == metric_id,
            MetricRanking.wallet_id == wallet_id,
            MetricRanking.time_period == time_period,
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def get_top_n(
        self,
        metric_id: int,
        n: int = None,
        time_period: str = TIME_PERIOD_ALL,
    ) -> List[MetricRanking]:
        """Get top N rankings for a metric for a time period"""
        n = n or self.top_n
        stmt = (
            select(MetricRanking)
            .where(
                MetricRanking.metric_id == metric_id,
                MetricRanking.time_period == time_period,
            )
            .order_by(MetricRanking.rank)
            .limit(n)
        )
        return list(self.session.execute(stmt).scalars().all())

    def get_top_n_with_wallets(
        self,
        metric_id: int,
        n: int = None,
        time_period: str = TIME_PERIOD_ALL,
    ) -> List[Tuple[MetricRanking, Wallet]]:
        """Get top N rankings with wallet info for a time period"""
        n = n or self.top_n
        stmt = (
            select(MetricRanking, Wallet)
            .join(Wallet, MetricRanking.wallet_id == Wallet.id)
            .where(
                MetricRanking.metric_id == metric_id,
                MetricRanking.time_period == time_period,
            )
            .order_by(MetricRanking.rank)
            .limit(n)
        )
        return list(self.session.execute(stmt).all())

    def get_cutoff_value(
        self,
        metric_id: int,
        time_period: str = TIME_PERIOD_ALL,
    ) -> Optional[float]:
        """Get the value at rank 100 (cutoff for inclusion) for a time period"""
        stmt = (
            select(MetricRanking.value)
            .where(
                MetricRanking.metric_id == metric_id,
                MetricRanking.time_period == time_period,
                MetricRanking.rank == self.top_n,
            )
        )
        return self.session.execute(stmt).scalar()

    def get_lowest_ranked(
        self,
        metric_id: int,
        time_period: str = TIME_PERIOD_ALL,
    ) -> Optional[MetricRanking]:
        """Get the lowest ranked entry for a metric for a time period"""
        stmt = (
            select(MetricRanking)
            .where(
                MetricRanking.metric_id == metric_id,
                MetricRanking.time_period == time_period,
            )
            .order_by(MetricRanking.rank.desc())
            .limit(1)
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def count_rankings(
        self,
        metric_id: int,
        time_period: str = TIME_PERIOD_ALL,
    ) -> int:
        """Count current rankings for a metric for a time period"""
        stmt = select(func.count(MetricRanking.id)).where(
            MetricRanking.metric_id == metric_id,
            MetricRanking.time_period == time_period,
        )
        return self.session.execute(stmt).scalar() or 0

    def qualifies_for_ranking(
        self,
        metric_id: int,
        value: float,
        higher_is_better: bool = True,
        time_period: str = TIME_PERIOD_ALL,
    ) -> bool:
        """Check if a value qualifies for Top 100 for a time period"""
        current_count = self.count_rankings(metric_id, time_period)

        # Always qualifies if less than 100 entries
        if current_count < self.top_n:
            return True

        # Get cutoff value
        cutoff = self.get_cutoff_value(metric_id, time_period)
        if cutoff is None:
            return True

        # Compare based on metric direction
        if higher_is_better:
            return value > cutoff
        else:
            return value < cutoff

    def upsert_ranking(
        self,
        metric_id: int,
        wallet_id: int,
        value: float,
        higher_is_better: bool = True,
        time_period: str = TIME_PERIOD_ALL,
    ) -> Optional[MetricRanking]:
        """
        Insert or update a ranking for a time period, maintaining sorted order.

        Returns the ranking if inserted/updated, None if not qualified.
        """
        # Check if wallet already has a ranking for this metric and time period
        existing = self.get_wallet_ranking(metric_id, wallet_id, time_period)

        if existing:
            # Update existing value
            existing.value = value
            existing.updated_at = datetime.utcnow()
            self.session.commit()
            # Reorder rankings
            self._reorder_rankings(metric_id, higher_is_better, time_period)
            return existing

        # Check if qualifies for ranking
        if not self.qualifies_for_ranking(metric_id, value, higher_is_better, time_period):
            return None

        # Find insert position
        rankings = self.get_top_n(metric_id, n=self.buffer_n, time_period=time_period)

        # Determine position based on value
        insert_pos = 1
        for ranking in rankings:
            if higher_is_better:
                if value > ranking.value:
                    break
            else:
                if value < ranking.value:
                    break
            insert_pos = ranking.rank + 1

        # Shift rankings down
        for ranking in reversed(rankings):
            if ranking.rank >= insert_pos:
                if ranking.rank < self.buffer_n:
                    ranking.rank += 1

        # Insert new ranking
        new_ranking = MetricRanking(
            metric_id=metric_id,
            wallet_id=wallet_id,
            time_period=time_period,
            rank=insert_pos,
            value=value,
            updated_at=datetime.utcnow(),
        )
        self.session.add(new_ranking)
        self.session.commit()

        # Trim to top N
        self._trim_rankings(metric_id, time_period)

        return new_ranking

    def _reorder_rankings(
        self,
        metric_id: int,
        higher_is_better: bool = True,
        time_period: str = TIME_PERIOD_ALL,
    ):
        """Reorder rankings after value update for a time period"""
        rankings = self.get_top_n(metric_id, n=self.buffer_n, time_period=time_period)

        # Sort by value
        if higher_is_better:
            rankings.sort(key=lambda r: r.value or 0, reverse=True)
        else:
            rankings.sort(key=lambda r: r.value or 0)

        # Reassign ranks
        for i, ranking in enumerate(rankings, 1):
            ranking.rank = i

        self.session.commit()
        self._trim_rankings(metric_id, time_period)

    def _trim_rankings(self, metric_id: int, time_period: str = TIME_PERIOD_ALL):
        """Remove rankings beyond top N for a time period"""
        stmt = delete(MetricRanking).where(
            MetricRanking.metric_id == metric_id,
            MetricRanking.time_period == time_period,
            MetricRanking.rank > self.top_n,
        )
        self.session.execute(stmt)
        self.session.commit()

    def rebuild_rankings(
        self,
        metric_id: int,
        higher_is_better: bool = True,
        time_period: str = TIME_PERIOD_ALL,
    ) -> int:
        """
        Rebuild rankings from wallet_metrics table for a time period.

        This is used for daily full recalculation.
        """
        # Delete existing rankings for this metric and time period
        stmt = delete(MetricRanking).where(
            MetricRanking.metric_id == metric_id,
            MetricRanking.time_period == time_period,
        )
        self.session.execute(stmt)

        # Get top values from wallet_metrics for this time period
        if higher_is_better:
            order_by = WalletMetric.value.desc()
        else:
            order_by = WalletMetric.value.asc()

        stmt = (
            select(WalletMetric)
            .where(
                WalletMetric.metric_id == metric_id,
                WalletMetric.time_period == time_period,
                WalletMetric.value.isnot(None),
            )
            .order_by(order_by)
            .limit(self.top_n)
        )
        top_wallet_metrics = list(self.session.execute(stmt).scalars().all())

        # Create new rankings
        for rank, wm in enumerate(top_wallet_metrics, 1):
            ranking = MetricRanking(
                metric_id=metric_id,
                wallet_id=wm.wallet_id,
                time_period=time_period,
                rank=rank,
                value=wm.value,
                percentile=wm.percentile,
                updated_at=datetime.utcnow(),
            )
            self.session.add(ranking)

        self.session.commit()
        return len(top_wallet_metrics)

    def rebuild_all_rankings(self) -> Dict[str, Dict[int, int]]:
        """Rebuild rankings for all metrics across all time periods"""
        from src.database.repositories.metric_repository import MetricRepository

        results = {}
        with MetricRepository(self.session) as metric_repo:
            metrics = metric_repo.get_all()

            for time_period in TIME_PERIODS:
                results[time_period] = {}
                for metric in metrics:
                    count = self.rebuild_rankings(
                        metric.id,
                        metric.higher_is_better,
                        time_period,
                    )
                    results[time_period][metric.id] = count

        return results

    def rebuild_rankings_for_period(
        self,
        time_period: str = TIME_PERIOD_ALL,
    ) -> Dict[int, int]:
        """Rebuild rankings for all metrics for a specific time period"""
        from src.database.repositories.metric_repository import MetricRepository

        results = {}
        with MetricRepository(self.session) as metric_repo:
            metrics = metric_repo.get_all()

            for metric in metrics:
                count = self.rebuild_rankings(
                    metric.id,
                    metric.higher_is_better,
                    time_period,
                )
                results[metric.id] = count

        return results

    def get_wallet_rankings(
        self,
        wallet_id: int,
        time_period: str = TIME_PERIOD_ALL,
    ) -> List[MetricRanking]:
        """Get all rankings for a wallet across all metrics for a time period"""
        stmt = (
            select(MetricRanking)
            .where(
                MetricRanking.wallet_id == wallet_id,
                MetricRanking.time_period == time_period,
            )
            .order_by(MetricRanking.metric_id)
        )
        return list(self.session.execute(stmt).scalars().all())

    def get_wallet_top_rankings(
        self,
        wallet_id: int,
        max_rank: int = 10,
        time_period: str = TIME_PERIOD_ALL,
    ) -> List[Tuple[MetricRanking, Metric]]:
        """Get wallet's best rankings (top 10 or better) for a time period"""
        stmt = (
            select(MetricRanking, Metric)
            .join(Metric, MetricRanking.metric_id == Metric.id)
            .where(
                MetricRanking.wallet_id == wallet_id,
                MetricRanking.time_period == time_period,
                MetricRanking.rank <= max_rank,
            )
            .order_by(MetricRanking.rank)
        )
        return list(self.session.execute(stmt).all())

    def get_ranking_summary(self, time_period: str = None) -> Dict[str, Any]:
        """Get summary statistics for rankings (optionally filtered by time period)"""
        # Total rankings
        total_stmt = select(func.count(MetricRanking.id))
        if time_period:
            total_stmt = total_stmt.where(MetricRanking.time_period == time_period)
        total = self.session.execute(total_stmt).scalar() or 0

        # Rankings per metric and time period
        per_metric_stmt = (
            select(
                MetricRanking.metric_id,
                MetricRanking.time_period,
                func.count(MetricRanking.id),
            )
            .group_by(MetricRanking.metric_id, MetricRanking.time_period)
        )
        if time_period:
            per_metric_stmt = per_metric_stmt.where(MetricRanking.time_period == time_period)
        per_metric_results = self.session.execute(per_metric_stmt).all()

        per_metric = {}
        for metric_id, period, count in per_metric_results:
            if period not in per_metric:
                per_metric[period] = {}
            per_metric[period][metric_id] = count

        # Unique wallets in rankings
        unique_wallets_stmt = select(
            func.count(func.distinct(MetricRanking.wallet_id))
        )
        if time_period:
            unique_wallets_stmt = unique_wallets_stmt.where(MetricRanking.time_period == time_period)
        unique_wallets = self.session.execute(unique_wallets_stmt).scalar() or 0

        return {
            "total_rankings": total,
            "rankings_per_metric": per_metric,
            "unique_wallets": unique_wallets,
            "expected_total": 117 * 100 * 3,  # 117 metrics × 100 ranks × 3 time periods
            "time_periods": len(TIME_PERIODS),
        }

    def clear_metric_rankings(
        self,
        metric_id: int,
        time_period: str = None,
    ) -> int:
        """Clear rankings for a metric (optionally for a specific time period)"""
        if time_period:
            count = self.count_rankings(metric_id, time_period)
            stmt = delete(MetricRanking).where(
                MetricRanking.metric_id == metric_id,
                MetricRanking.time_period == time_period,
            )
        else:
            # Clear all time periods for this metric
            stmt = select(func.count(MetricRanking.id)).where(
                MetricRanking.metric_id == metric_id
            )
            count = self.session.execute(stmt).scalar() or 0
            stmt = delete(MetricRanking).where(MetricRanking.metric_id == metric_id)
        self.session.execute(stmt)
        self.session.commit()
        return count

    def clear_all_rankings(self, time_period: str = None) -> int:
        """Clear all rankings (optionally for a specific time period)"""
        if time_period:
            stmt = select(func.count(MetricRanking.id)).where(
                MetricRanking.time_period == time_period
            )
            count = self.session.execute(stmt).scalar() or 0
            stmt = delete(MetricRanking).where(MetricRanking.time_period == time_period)
        else:
            stmt = select(func.count(MetricRanking.id))
            count = self.session.execute(stmt).scalar() or 0
            stmt = delete(MetricRanking)
        self.session.execute(stmt)
        self.session.commit()
        return count
