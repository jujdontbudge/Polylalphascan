"""Repository for metric data access"""
import yaml
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy import select, func
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from src.database.models import Metric, WalletMetric, get_session, TIME_PERIOD_ALL, TIME_PERIODS
from config.settings import CONFIG_DIR


class MetricRepository:
    """Data access layer for metrics"""

    def __init__(self, session: Optional[Session] = None):
        self.session = session or get_session()
        self._owns_session = session is None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._owns_session:
            self.session.close()

    def get_by_id(self, metric_id: int) -> Optional[Metric]:
        """Get metric by ID"""
        return self.session.get(Metric, metric_id)

    def get_by_code(self, code: str) -> Optional[Metric]:
        """Get metric by code"""
        stmt = select(Metric).where(Metric.code == code)
        return self.session.execute(stmt).scalar_one_or_none()

    def get_all(self) -> List[Metric]:
        """Get all metrics"""
        stmt = select(Metric).order_by(Metric.id)
        return list(self.session.execute(stmt).scalars().all())

    def get_by_group(self, group_id: str) -> List[Metric]:
        """Get metrics by group ID"""
        stmt = select(Metric).where(Metric.group_id == group_id).order_by(Metric.id)
        return list(self.session.execute(stmt).scalars().all())

    def count(self) -> int:
        """Count total metrics"""
        stmt = select(func.count(Metric.id))
        return self.session.execute(stmt).scalar() or 0

    def load_from_yaml(self, yaml_path: Path = None) -> List[Metric]:
        """Load metrics from YAML configuration"""
        if yaml_path is None:
            yaml_path = CONFIG_DIR / "metrics_config.yaml"

        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        metrics = []
        for m in config.get('metrics', []):
            metric = Metric(
                id=m['id'],
                group_id=m['group_id'],
                code=m['code'],
                name=m['name'],
                description=m.get('description', ''),
                higher_is_better=m.get('higher_is_better', True),
                formula=m.get('formula', ''),
            )
            metrics.append(metric)

        return metrics

    def initialize_metrics(self, yaml_path: Path = None) -> int:
        """Initialize metric definitions from YAML"""
        metrics = self.load_from_yaml(yaml_path)

        count = 0
        for metric in metrics:
            existing = self.get_by_id(metric.id)
            if not existing:
                self.session.add(metric)
                count += 1
            else:
                # Update existing metric
                existing.group_id = metric.group_id
                existing.code = metric.code
                existing.name = metric.name
                existing.description = metric.description
                existing.higher_is_better = metric.higher_is_better
                existing.formula = metric.formula

        self.session.commit()
        return count

    def get_metric_ids_by_group(self) -> Dict[str, List[int]]:
        """Get metric IDs organized by group"""
        metrics = self.get_all()
        groups = {}
        for metric in metrics:
            if metric.group_id not in groups:
                groups[metric.group_id] = []
            groups[metric.group_id].append(metric.id)
        return groups

    # Wallet metric operations

    def set_wallet_metric(
        self,
        wallet_id: int,
        metric_id: int,
        value: float,
        percentile: float = None,
        time_period: str = TIME_PERIOD_ALL,
    ) -> WalletMetric:
        """Set or update a wallet metric value for a specific time period"""
        stmt = select(WalletMetric).where(
            WalletMetric.wallet_id == wallet_id,
            WalletMetric.metric_id == metric_id,
            WalletMetric.time_period == time_period,
        )
        wallet_metric = self.session.execute(stmt).scalar_one_or_none()

        if wallet_metric:
            wallet_metric.value = value
            wallet_metric.percentile = percentile
            wallet_metric.calculated_at = datetime.utcnow()
        else:
            wallet_metric = WalletMetric(
                wallet_id=wallet_id,
                metric_id=metric_id,
                time_period=time_period,
                value=value,
                percentile=percentile,
                calculated_at=datetime.utcnow(),
            )
            self.session.add(wallet_metric)

        self.session.commit()
        return wallet_metric

    def bulk_set_wallet_metrics(
        self,
        wallet_id: int,
        metrics: Dict[int, float],
        percentiles: Dict[int, float] = None,
        time_period: str = TIME_PERIOD_ALL,
    ) -> int:
        """Bulk set wallet metrics for a specific time period using batch upsert"""
        percentiles = percentiles or {}

        # Filter out None values
        valid_metrics = {k: v for k, v in metrics.items() if v is not None}
        if not valid_metrics:
            return 0

        # Build batch insert data
        now = datetime.utcnow()
        records = [
            {
                "wallet_id": wallet_id,
                "metric_id": metric_id,
                "time_period": time_period,
                "value": value,
                "percentile": percentiles.get(metric_id),
                "calculated_at": now,
            }
            for metric_id, value in valid_metrics.items()
        ]

        # Use INSERT...ON CONFLICT for batch upsert
        stmt = insert(WalletMetric).values(records)
        stmt = stmt.on_conflict_do_update(
            index_elements=["wallet_id", "metric_id", "time_period"],
            set_={
                "value": stmt.excluded.value,
                "percentile": stmt.excluded.percentile,
                "calculated_at": stmt.excluded.calculated_at,
            },
        )

        self.session.execute(stmt)
        self.session.commit()
        return len(records)

    def batch_set_multiple_wallets_metrics(
        self,
        wallet_metrics: List[Dict[str, Any]],
    ) -> int:
        """
        Batch set metrics for multiple wallets at once.

        Args:
            wallet_metrics: List of dicts with:
                - wallet_id: int
                - metrics: Dict[int, float] (metric_id -> value)
                - time_period: str (optional, defaults to all_time)

        Returns:
            Total records upserted
        """
        if not wallet_metrics:
            return 0

        now = datetime.utcnow()
        all_records = []

        for entry in wallet_metrics:
            wallet_id = entry["wallet_id"]
            metrics = entry.get("metrics", {})
            time_period = entry.get("time_period", TIME_PERIOD_ALL)
            percentiles = entry.get("percentiles", {})

            for metric_id, value in metrics.items():
                if value is not None:
                    all_records.append({
                        "wallet_id": wallet_id,
                        "metric_id": metric_id,
                        "time_period": time_period,
                        "value": value,
                        "percentile": percentiles.get(metric_id),
                        "calculated_at": now,
                    })

        if not all_records:
            return 0

        # Process in batches of 1000 for efficiency
        batch_size = 1000
        total = 0

        for i in range(0, len(all_records), batch_size):
            batch = all_records[i:i + batch_size]
            stmt = insert(WalletMetric).values(batch)
            stmt = stmt.on_conflict_do_update(
                index_elements=["wallet_id", "metric_id", "time_period"],
                set_={
                    "value": stmt.excluded.value,
                    "percentile": stmt.excluded.percentile,
                    "calculated_at": stmt.excluded.calculated_at,
                },
            )
            self.session.execute(stmt)
            total += len(batch)

        self.session.commit()
        return total

    def get_wallet_metrics(
        self,
        wallet_id: int,
        metric_ids: List[int] = None,
        time_period: str = TIME_PERIOD_ALL,
    ) -> Dict[int, float]:
        """Get all metric values for a wallet for a specific time period"""
        stmt = select(WalletMetric).where(
            WalletMetric.wallet_id == wallet_id,
            WalletMetric.time_period == time_period,
        )
        if metric_ids:
            stmt = stmt.where(WalletMetric.metric_id.in_(metric_ids))

        results = self.session.execute(stmt).scalars().all()
        return {wm.metric_id: wm.value for wm in results}

    def get_wallet_metric(
        self,
        wallet_id: int,
        metric_id: int,
        time_period: str = TIME_PERIOD_ALL,
    ) -> Optional[float]:
        """Get single metric value for a wallet for a specific time period"""
        stmt = select(WalletMetric.value).where(
            WalletMetric.wallet_id == wallet_id,
            WalletMetric.metric_id == metric_id,
            WalletMetric.time_period == time_period,
        )
        return self.session.execute(stmt).scalar()

    def get_metric_values(
        self,
        metric_id: int,
        time_period: str = TIME_PERIOD_ALL,
    ) -> List[Dict[str, Any]]:
        """Get all values for a metric across wallets for a specific time period"""
        stmt = (
            select(
                WalletMetric.wallet_id,
                WalletMetric.value,
                WalletMetric.percentile,
            )
            .where(
                WalletMetric.metric_id == metric_id,
                WalletMetric.time_period == time_period,
            )
            .order_by(WalletMetric.value.desc())
        )
        results = self.session.execute(stmt).all()
        return [
            {"wallet_id": r[0], "value": r[1], "percentile": r[2]}
            for r in results
        ]

    def calculate_percentiles(
        self,
        metric_id: int,
        time_period: str = TIME_PERIOD_ALL,
    ) -> int:
        """Calculate and update percentiles for a metric for a specific time period"""
        metric = self.get_by_id(metric_id)
        if not metric:
            return 0

        # Get all values for this time period
        stmt = (
            select(WalletMetric)
            .where(
                WalletMetric.metric_id == metric_id,
                WalletMetric.time_period == time_period,
            )
            .where(WalletMetric.value.isnot(None))
        )
        wallet_metrics = list(self.session.execute(stmt).scalars().all())

        if not wallet_metrics:
            return 0

        # Sort by value
        if metric.higher_is_better:
            wallet_metrics.sort(key=lambda x: x.value or 0, reverse=True)
        else:
            wallet_metrics.sort(key=lambda x: x.value or 0)

        # Calculate percentiles
        total = len(wallet_metrics)
        for i, wm in enumerate(wallet_metrics):
            wm.percentile = ((total - i) / total) * 100

        self.session.commit()
        return total

    def calculate_all_percentiles(self, metric_id: int) -> Dict[str, int]:
        """Calculate percentiles for all time periods for a metric"""
        results = {}
        for time_period in TIME_PERIODS:
            results[time_period] = self.calculate_percentiles(metric_id, time_period)
        return results

    def delete_wallet_metrics(self, wallet_id: int) -> int:
        """Delete all metrics for a wallet"""
        stmt = select(WalletMetric).where(WalletMetric.wallet_id == wallet_id)
        wallet_metrics = list(self.session.execute(stmt).scalars().all())

        count = len(wallet_metrics)
        for wm in wallet_metrics:
            self.session.delete(wm)

        self.session.commit()
        return count
