"""Excel exporter for PolyScan data"""
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

from src.database.repositories import WalletRepository, MetricRepository, RankingRepository
from src.database.models import get_session
from src.core.classifier import CopyRosterGenerator
from src.core.ranker import RankingEngine
from config.settings import EXPORTS_DIR

logger = logging.getLogger(__name__)


class ExcelExporter:
    """Export PolyScan data to Excel files"""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or EXPORTS_DIR
        self.output_dir.mkdir(exist_ok=True)

    def export_leaderboards(
        self,
        filename: str = None,
        metric_ids: List[int] = None,
    ) -> Path:
        """
        Export Top 100 leaderboards for all or selected metrics.

        Args:
            filename: Output filename (default: auto-generated)
            metric_ids: Specific metrics to export (default: all)

        Returns:
            Path to exported file
        """
        filename = filename or f"leaderboards_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        filepath = self.output_dir / filename

        logger.info(f"Exporting leaderboards to {filepath}")

        session = get_session()
        try:
            with MetricRepository(session) as metric_repo:
                metrics = metric_repo.get_all()

            if metric_ids:
                metrics = [m for m in metrics if m.id in metric_ids]

            ranking_engine = RankingEngine()

            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Summary sheet
                summary_data = {
                    "metric_id": [],
                    "metric_code": [],
                    "metric_name": [],
                    "group": [],
                    "top_wallet": [],
                    "top_value": [],
                }

                used_sheet_names = set()

                for metric in metrics:
                    leaderboard = ranking_engine.get_leaderboard(metric.id, limit=100)

                    if not leaderboard:
                        continue

                    # Add to summary
                    summary_data["metric_id"].append(metric.id)
                    summary_data["metric_code"].append(metric.code)
                    summary_data["metric_name"].append(metric.name)
                    summary_data["group"].append(metric.group_id)
                    summary_data["top_wallet"].append(leaderboard[0]["wallet_address"][:10] + "...")
                    summary_data["top_value"].append(leaderboard[0]["value"])

                    # Create sheet for this metric with collision avoidance
                    sheet_name = f"{metric.id}_{metric.code}"[:31]
                    if sheet_name in used_sheet_names:
                        suffix = 2
                        while True:
                            candidate = f"{metric.id}_{metric.code}"[:28] + f"_{suffix}"
                            if candidate not in used_sheet_names:
                                sheet_name = candidate
                                break
                            suffix += 1
                        logger.warning(f"Sheet name collision for metric {metric.id}, using '{sheet_name}'")
                    used_sheet_names.add(sheet_name)

                    df = pd.DataFrame(leaderboard)
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

                # Write summary sheet first
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name="Summary", index=False)

        finally:
            session.close()

        logger.info(f"Exported {len(metrics)} leaderboards to {filepath}")
        return filepath

    def export_copy_roster(
        self,
        filename: str = None,
        limit: int = 50,
    ) -> Path:
        """
        Export copy-trading roster to Excel.

        Args:
            filename: Output filename
            limit: Maximum candidates

        Returns:
            Path to exported file
        """
        filename = filename or f"copy_roster_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        filepath = self.output_dir / filename

        logger.info(f"Exporting copy roster to {filepath}")

        generator = CopyRosterGenerator()
        roster = generator.generate_roster(limit=limit)
        roster_data = generator.export_roster(roster, format="dict")

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Main roster
            main_data = []
            for c in roster_data:
                main_data.append({
                    "Rank": len(main_data) + 1,
                    "Address": c["address"],
                    "Classification": c["classification"],
                    "Copy Score": round(c["copy_score"], 1),
                    "Quant Score": round(c["quant_score"], 1),
                    "Performance": round(c["performance_score"], 1),
                    "Risk": round(c["risk_score"], 1),
                    "Copyability": round(c["copyability_score"], 1),
                    "ROI %": round(c["key_metrics"].get("roi", 0), 2),
                    "Sharpe": round(c["key_metrics"].get("sharpe_ratio", 0), 2),
                    "Win Rate %": round(c["key_metrics"].get("win_rate", 0), 1),
                    "Max DD %": round(c["key_metrics"].get("max_drawdown", 0), 1),
                    "Recommendation": c["recommendation"],
                })

            df = pd.DataFrame(main_data)
            df.to_excel(writer, sheet_name="Copy Roster", index=False)

            # Top rankings detail
            rankings_data = []
            for c in roster_data:
                for ranking in c.get("top_rankings", []):
                    rankings_data.append({
                        "Address": c["address"][:10] + "...",
                        "Metric": ranking["metric_name"],
                        "Rank": ranking["rank"],
                        "Value": ranking["value"],
                    })

            if rankings_data:
                rankings_df = pd.DataFrame(rankings_data)
                rankings_df.to_excel(writer, sheet_name="Top Rankings", index=False)

            # Summary
            summary = generator.get_roster_summary(roster)
            summary_df = pd.DataFrame([summary])
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

        logger.info(f"Exported {len(roster)} candidates to {filepath}")
        return filepath

    def export_wallet_profile(
        self,
        wallet_address: str,
        filename: str = None,
    ) -> Optional[Path]:
        """
        Export detailed profile for a single wallet.

        Args:
            wallet_address: Wallet address
            filename: Output filename

        Returns:
            Path to exported file or None
        """
        session = get_session()
        try:
            with WalletRepository(session) as wallet_repo, \
                 MetricRepository(session) as metric_repo:

                wallet = wallet_repo.get_by_address(wallet_address)
                if not wallet:
                    logger.error(f"Wallet not found: {wallet_address}")
                    return None

                metrics = metric_repo.get_wallet_metrics(wallet.id)
                all_metrics = metric_repo.get_all()
                metrics_map = {m.id: m for m in all_metrics}

        finally:
            session.close()

        filename = filename or f"wallet_{wallet_address[:8]}_{datetime.now().strftime('%Y%m%d')}.xlsx"
        filepath = self.output_dir / filename

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Wallet info
            info_data = {
                "Field": ["Address", "Classification", "Quant Score", "Total Volume",
                          "Total Trades", "First Seen", "Last Scanned"],
                "Value": [
                    wallet.address,
                    wallet.classification,
                    wallet.quant_score,
                    wallet.total_volume,
                    wallet.total_trades,
                    wallet.first_seen_at.isoformat() if wallet.first_seen_at else "N/A",
                    wallet.last_scanned_at.isoformat() if wallet.last_scanned_at else "N/A",
                ],
            }
            pd.DataFrame(info_data).to_excel(writer, sheet_name="Info", index=False)

            # All metrics by group
            for group_id in "ABCDEFGHIJKLMNO":
                group_metrics = [m for m in all_metrics if m.group_id == group_id]
                if not group_metrics:
                    continue

                group_data = []
                for metric in group_metrics:
                    value = metrics.get(metric.id)
                    group_data.append({
                        "ID": metric.id,
                        "Code": metric.code,
                        "Name": metric.name,
                        "Value": round(value, 4) if value is not None else "N/A",
                        "Higher is Better": "Yes" if metric.higher_is_better else "No",
                    })

                df = pd.DataFrame(group_data)
                df.to_excel(writer, sheet_name=f"Group {group_id}", index=False)

        logger.info(f"Exported wallet profile to {filepath}")
        return filepath

    def export_all_wallets(
        self,
        filename: str = None,
        include_metrics: bool = False,
    ) -> Path:
        """
        Export all wallets to Excel.

        Args:
            filename: Output filename
            include_metrics: Whether to include metric values

        Returns:
            Path to exported file
        """
        filename = filename or f"all_wallets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        filepath = self.output_dir / filename

        session = get_session()
        try:
            with WalletRepository(session) as wallet_repo:
                wallets = wallet_repo.get_all()

            wallet_data = []
            for w in wallets:
                wallet_data.append({
                    "ID": w.id,
                    "Address": w.address,
                    "Classification": w.classification,
                    "Quant Score": w.quant_score,
                    "Total Volume": w.total_volume,
                    "Total Trades": w.total_trades,
                    "Total PnL": w.total_pnl,
                    "First Seen": w.first_seen_at.isoformat() if w.first_seen_at else None,
                    "Last Scanned": w.last_scanned_at.isoformat() if w.last_scanned_at else None,
                    "Elite Flag": w.is_elite_flag,
                })

        finally:
            session.close()

        df = pd.DataFrame(wallet_data)
        df.to_excel(filepath, index=False, sheet_name="Wallets")

        logger.info(f"Exported {len(wallets)} wallets to {filepath}")
        return filepath
