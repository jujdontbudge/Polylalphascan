"""JSON exporter for PolyScan data"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from src.database.repositories import WalletRepository, MetricRepository, RankingRepository
from src.database.models import get_session
from src.core.classifier import CopyRosterGenerator
from src.core.ranker import RankingEngine
from config.settings import EXPORTS_DIR

logger = logging.getLogger(__name__)


class JsonExporter:
    """Export PolyScan data to JSON files"""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or EXPORTS_DIR
        self.output_dir.mkdir(exist_ok=True)

    def _write_json(self, filepath: Path, data: Any) -> Path:
        """Write data to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return filepath

    def export_leaderboards(
        self,
        filename: str = None,
        metric_ids: List[int] = None,
    ) -> Path:
        """
        Export Top 100 leaderboards to JSON.

        Args:
            filename: Output filename
            metric_ids: Specific metrics to export

        Returns:
            Path to exported file
        """
        filename = filename or f"leaderboards_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.output_dir / filename

        logger.info(f"Exporting leaderboards to {filepath}")

        session = get_session()
        try:
            with MetricRepository(session) as metric_repo:
                metrics = metric_repo.get_all()

            if metric_ids:
                metrics = [m for m in metrics if m.id in metric_ids]

            ranking_engine = RankingEngine()

            data = {
                "generated_at": datetime.utcnow().isoformat(),
                "total_metrics": len(metrics),
                "leaderboards": {},
            }

            for metric in metrics:
                leaderboard = ranking_engine.get_leaderboard(metric.id, limit=100)

                data["leaderboards"][metric.code] = {
                    "metric_id": metric.id,
                    "metric_name": metric.name,
                    "group_id": metric.group_id,
                    "higher_is_better": metric.higher_is_better,
                    "rankings": leaderboard,
                }

        finally:
            session.close()

        self._write_json(filepath, data)
        logger.info(f"Exported {len(metrics)} leaderboards to {filepath}")
        return filepath

    def export_copy_roster(
        self,
        filename: str = None,
        limit: int = 50,
    ) -> Path:
        """
        Export copy-trading roster to JSON.

        Args:
            filename: Output filename
            limit: Maximum candidates

        Returns:
            Path to exported file
        """
        filename = filename or f"copy_roster_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.output_dir / filename

        logger.info(f"Exporting copy roster to {filepath}")

        generator = CopyRosterGenerator()
        roster = generator.generate_roster(limit=limit)
        roster_data = generator.export_roster(roster, format="dict")
        summary = generator.get_roster_summary(roster)

        data = {
            "generated_at": datetime.utcnow().isoformat(),
            "summary": summary,
            "candidates": roster_data,
        }

        self._write_json(filepath, data)
        logger.info(f"Exported {len(roster)} candidates to {filepath}")
        return filepath

    def export_wallet_profile(
        self,
        wallet_address: str,
        filename: str = None,
    ) -> Optional[Path]:
        """
        Export detailed wallet profile to JSON.

        Args:
            wallet_address: Wallet address
            filename: Output filename

        Returns:
            Path to exported file or None
        """
        session = get_session()
        try:
            with WalletRepository(session) as wallet_repo, \
                 MetricRepository(session) as metric_repo, \
                 RankingRepository(session) as ranking_repo:

                wallet = wallet_repo.get_by_address(wallet_address)
                if not wallet:
                    logger.error(f"Wallet not found: {wallet_address}")
                    return None

                metrics = metric_repo.get_wallet_metrics(wallet.id)
                all_metrics = metric_repo.get_all()
                rankings = ranking_repo.get_wallet_rankings(wallet.id)

        finally:
            session.close()

        filename = filename or f"wallet_{wallet_address[:8]}_{datetime.now().strftime('%Y%m%d')}.json"
        filepath = self.output_dir / filename

        metrics_map = {m.id: m for m in all_metrics}

        data = {
            "generated_at": datetime.utcnow().isoformat(),
            "wallet": {
                "id": wallet.id,
                "address": wallet.address,
                "classification": wallet.classification,
                "quant_score": wallet.quant_score,
                "total_volume": wallet.total_volume,
                "total_trades": wallet.total_trades,
                "total_pnl": wallet.total_pnl,
                "first_seen_at": wallet.first_seen_at.isoformat() if wallet.first_seen_at else None,
                "last_scanned_at": wallet.last_scanned_at.isoformat() if wallet.last_scanned_at else None,
                "is_elite_flag": wallet.is_elite_flag,
            },
            "metrics": {
                metrics_map[mid].code: {
                    "metric_id": mid,
                    "name": metrics_map[mid].name,
                    "group_id": metrics_map[mid].group_id,
                    "value": value,
                    "higher_is_better": metrics_map[mid].higher_is_better,
                }
                for mid, value in metrics.items()
                if mid in metrics_map
            },
            "rankings": [
                {
                    "metric_id": r.metric_id,
                    "metric_code": metrics_map[r.metric_id].code if r.metric_id in metrics_map else None,
                    "rank": r.rank,
                    "value": r.value,
                    "percentile": r.percentile,
                }
                for r in rankings
            ],
        }

        self._write_json(filepath, data)
        logger.info(f"Exported wallet profile to {filepath}")
        return filepath

    def export_all_wallets(
        self,
        filename: str = None,
    ) -> Path:
        """
        Export all wallets to JSON.

        Args:
            filename: Output filename

        Returns:
            Path to exported file
        """
        filename = filename or f"all_wallets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.output_dir / filename

        session = get_session()
        try:
            with WalletRepository(session) as wallet_repo:
                wallets = wallet_repo.get_all()

        finally:
            session.close()

        data = {
            "generated_at": datetime.utcnow().isoformat(),
            "total_wallets": len(wallets),
            "wallets": [
                {
                    "id": w.id,
                    "address": w.address,
                    "classification": w.classification,
                    "quant_score": w.quant_score,
                    "total_volume": w.total_volume,
                    "total_trades": w.total_trades,
                    "total_pnl": w.total_pnl,
                    "first_seen_at": w.first_seen_at.isoformat() if w.first_seen_at else None,
                    "last_scanned_at": w.last_scanned_at.isoformat() if w.last_scanned_at else None,
                    "is_elite_flag": w.is_elite_flag,
                }
                for w in wallets
            ],
        }

        self._write_json(filepath, data)
        logger.info(f"Exported {len(wallets)} wallets to {filepath}")
        return filepath

    def export_ranking_summary(
        self,
        filename: str = None,
    ) -> Path:
        """
        Export ranking summary statistics.

        Args:
            filename: Output filename

        Returns:
            Path to exported file
        """
        filename = filename or f"ranking_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.output_dir / filename

        ranking_engine = RankingEngine()
        summary = ranking_engine.get_ranking_summary()

        data = {
            "generated_at": datetime.utcnow().isoformat(),
            **summary,
        }

        self._write_json(filepath, data)
        logger.info(f"Exported ranking summary to {filepath}")
        return filepath

    def export_metrics_definition(
        self,
        filename: str = None,
    ) -> Path:
        """
        Export all 117 metric definitions.

        Args:
            filename: Output filename

        Returns:
            Path to exported file
        """
        filename = filename or "metrics_definition.json"
        filepath = self.output_dir / filename

        session = get_session()
        try:
            with MetricRepository(session) as metric_repo:
                metrics = metric_repo.get_all()

        finally:
            session.close()

        data = {
            "total_metrics": len(metrics),
            "groups": {},
            "metrics": {},
        }

        for metric in metrics:
            # Add to metrics dict
            data["metrics"][metric.code] = {
                "id": metric.id,
                "group_id": metric.group_id,
                "name": metric.name,
                "description": metric.description,
                "higher_is_better": metric.higher_is_better,
                "formula": metric.formula,
            }

            # Add to groups summary
            if metric.group_id not in data["groups"]:
                data["groups"][metric.group_id] = {
                    "metric_ids": [],
                    "metric_codes": [],
                }
            data["groups"][metric.group_id]["metric_ids"].append(metric.id)
            data["groups"][metric.group_id]["metric_codes"].append(metric.code)

        self._write_json(filepath, data)
        logger.info(f"Exported {len(metrics)} metric definitions to {filepath}")
        return filepath
