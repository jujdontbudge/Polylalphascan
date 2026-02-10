"""
Copy Roster Generator - Creates curated list of wallets suitable for copy-trading.

Combines multiple factors:
1. Classification (bot/quant preferred)
2. Performance metrics (profitability, Sharpe, win rate)
3. Copy-ability metrics (frequency, scalability, transparency)
4. Ranking positions across metrics
"""
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from src.database.repositories import WalletRepository, MetricRepository, RankingRepository
from src.database.models import get_session, Wallet
from src.core.ranker import RankingEngine

logger = logging.getLogger(__name__)


@dataclass
class CopyCandidate:
    """A wallet candidate for copy-trading"""
    wallet_id: int
    address: str
    classification: str
    quant_score: float
    copy_score: float  # 0-100 overall suitability
    performance_score: float
    risk_score: float
    copyability_score: float
    top_rankings: List[Dict[str, Any]]
    key_metrics: Dict[str, float]
    recommendation: str


class CopyRosterGenerator:
    """
    Generates curated list of wallets for copy-trading.

    Scoring factors:
    - Performance (40%): ROI, Sharpe, win rate
    - Copyability (30%): Frequency, scalability, transparency
    - Consistency (20%): Strategy consistency, discipline
    - Bot signals (10%): Higher quant score = bonus

    Filters:
    - Must be bot or quant classification
    - Minimum number of rankings (shows consistent performance)
    - Minimum trading history
    """

    # Key metrics for scoring
    PERFORMANCE_METRICS = {
        2: ("total_roi", 0.3),
        11: ("sharpe_ratio", 0.3),
        22: ("win_rate", 0.2),
        4: ("profit_factor", 0.2),
    }

    RISK_METRICS = {
        14: ("max_drawdown", -0.4),  # Lower is better
        17: ("volatility", -0.3),     # Lower is better
        21: ("recovery_factor", 0.3),
    }

    COPYABILITY_METRICS = {
        95: ("trade_frequency_score", 0.3),
        98: ("copy_scalability", 0.3),
        99: ("transparency_score", 0.2),
        100: ("risk_adjusted_copyability", 0.2),
    }

    CONSISTENCY_METRICS = {
        66: ("strategy_consistency", 0.5),
        74: ("discipline_score", 0.5),
    }

    def __init__(self):
        self.ranking_engine = RankingEngine()

    def generate_roster(
        self,
        limit: int = 50,
        min_rankings: int = 3,
        classifications: List[str] = None,
    ) -> List[CopyCandidate]:
        """
        Generate copy-trading roster.

        Args:
            limit: Maximum candidates to return
            min_rankings: Minimum Top 100 appearances
            classifications: Accepted classifications (default: bot, quant)

        Returns:
            List of CopyCandidate objects, sorted by copy_score
        """
        classifications = classifications or ["bot", "quant"]

        logger.info(f"Generating copy roster (limit={limit}, classifications={classifications})")

        # Get candidates with multiple rankings
        elite_wallets = self.ranking_engine.get_top_wallets_across_metrics(
            min_rankings=min_rankings,
            max_avg_rank=50,
        )

        candidates = []

        for wallet_info in elite_wallets:
            # Filter by classification
            if wallet_info["classification"] not in classifications:
                continue

            # Create full candidate profile
            candidate = self._create_candidate(wallet_info)
            if candidate:
                candidates.append(candidate)

        # Sort by copy_score
        candidates.sort(key=lambda c: c.copy_score, reverse=True)

        return candidates[:limit]

    def _create_candidate(self, wallet_info: Dict[str, Any]) -> Optional[CopyCandidate]:
        """Create a full CopyCandidate from wallet info"""
        wallet_id = wallet_info["wallet_id"]

        session = get_session()
        try:
            with MetricRepository(session) as metric_repo, \
                 RankingRepository(session) as ranking_repo:

                # Get all metrics
                metrics = metric_repo.get_wallet_metrics(wallet_id)

                if not metrics:
                    return None

                # Get top rankings
                top_rankings_raw = ranking_repo.get_wallet_top_rankings(wallet_id, max_rank=10)
                top_rankings = [
                    {
                        "metric_code": metric.code,
                        "metric_name": metric.name,
                        "rank": ranking.rank,
                        "value": ranking.value,
                    }
                    for ranking, metric in top_rankings_raw
                ]

        finally:
            session.close()

        # Calculate component scores
        performance_score = self._calculate_component_score(
            metrics, self.PERFORMANCE_METRICS
        )
        risk_score = self._calculate_component_score(
            metrics, self.RISK_METRICS
        )
        copyability_score = self._calculate_component_score(
            metrics, self.COPYABILITY_METRICS
        )
        consistency_score = self._calculate_component_score(
            metrics, self.CONSISTENCY_METRICS
        )

        # Calculate overall copy score
        copy_score = (
            performance_score * 0.35 +
            copyability_score * 0.25 +
            consistency_score * 0.15 +
            (100 - risk_score) * 0.15 +  # Lower risk is better
            wallet_info["quant_score"] * 0.10
        )

        # Extract key metrics
        key_metrics = {
            "roi": metrics.get(2, 0),
            "sharpe_ratio": metrics.get(11, 0),
            "win_rate": metrics.get(22, 0),
            "max_drawdown": metrics.get(14, 0),
            "trades_per_day": metrics.get(36, 0),
            "quant_score": wallet_info["quant_score"],
        }

        # Generate recommendation
        recommendation = self._generate_recommendation(
            wallet_info, copy_score, performance_score, risk_score
        )

        return CopyCandidate(
            wallet_id=wallet_id,
            address=wallet_info["address"],
            classification=wallet_info["classification"],
            quant_score=wallet_info["quant_score"],
            copy_score=copy_score,
            performance_score=performance_score,
            risk_score=risk_score,
            copyability_score=copyability_score,
            top_rankings=top_rankings,
            key_metrics=key_metrics,
            recommendation=recommendation,
        )

    # Expected value ranges per metric for proper normalization
    METRIC_RANGES = {
        2: (0, 200),       # ROI: 0-200%
        4: (0, 10),        # Profit factor: 0-10
        11: (-2, 5),       # Sharpe: -2 to 5
        14: (0, 100),      # Max drawdown: 0-100%
        17: (0, 200),      # Volatility: 0-200%
        21: (0, 10),       # Recovery factor: 0-10
        22: (0, 100),      # Win rate: 0-100%
        66: (0, 100),      # Strategy consistency: 0-100
        74: (0, 100),      # Discipline: 0-100
        95: (0, 100),      # Trade frequency score: 0-100
        98: (0, 100),      # Copy scalability: 0-100
        99: (0, 100),      # Transparency: 0-100
        100: (0, 100),     # Risk-adj copyability: 0-100
    }

    def _calculate_component_score(
        self,
        metrics: Dict[int, float],
        metric_weights: Dict[int, tuple],
    ) -> float:
        """Calculate weighted component score with proper range normalization"""
        total_weight = 0
        weighted_sum = 0
        missing_count = 0

        for metric_id, (name, weight) in metric_weights.items():
            if metric_id not in metrics or metrics[metric_id] is None:
                missing_count += 1
                logger.debug(f"Missing metric {metric_id} ({name}) for scoring")
                continue

            value = metrics[metric_id]

            # Normalize using known ranges
            min_range, max_range = self.METRIC_RANGES.get(metric_id, (0, 100))
            range_size = max_range - min_range
            if range_size > 0:
                normalized = (value - min_range) / range_size * 100
            else:
                normalized = value

            # Clip to [0, 100]
            normalized = max(0.0, min(100.0, normalized))

            # For negative weight (lower is better), invert
            if weight < 0:
                normalized = 100 - normalized

            weighted_sum += normalized * abs(weight)
            total_weight += abs(weight)

        if missing_count > 0:
            logger.debug(f"{missing_count} metrics missing from scoring component")

        if total_weight == 0:
            return 50.0  # Neutral default

        return weighted_sum / total_weight

    def _generate_recommendation(
        self,
        wallet_info: Dict[str, Any],
        copy_score: float,
        performance_score: float,
        risk_score: float,
    ) -> str:
        """Generate recommendation text"""
        stars = min(5, int(copy_score / 20) + 1)
        star_str = "*" * stars

        if copy_score >= 80:
            tier = "ELITE"
            advice = "Highest confidence for copy-trading. Excellent track record."
        elif copy_score >= 60:
            tier = "STRONG"
            advice = "Good candidate for copy-trading with solid performance."
        elif copy_score >= 40:
            tier = "MODERATE"
            advice = "Consider with smaller allocation. Monitor closely."
        else:
            tier = "WATCHLIST"
            advice = "Keep on watchlist. Needs more data or improvement."

        return (
            f"{star_str} {tier} "
            f"(Copy Score: {copy_score:.0f}, "
            f"Performance: {performance_score:.0f}, "
            f"Risk: {risk_score:.0f}) - {advice}"
        )

    def get_roster_summary(
        self,
        roster: List[CopyCandidate],
    ) -> Dict[str, Any]:
        """Generate summary statistics for a roster"""
        if not roster:
            return {"count": 0}

        return {
            "count": len(roster),
            "avg_copy_score": sum(c.copy_score for c in roster) / len(roster),
            "avg_quant_score": sum(c.quant_score for c in roster) / len(roster),
            "classification_breakdown": {
                classification: len([c for c in roster if c.classification == classification])
                for classification in set(c.classification for c in roster)
            },
            "top_performer": {
                "address": roster[0].address,
                "copy_score": roster[0].copy_score,
            } if roster else None,
            "generated_at": datetime.utcnow().isoformat(),
        }

    def export_roster(
        self,
        roster: List[CopyCandidate],
        format: str = "dict",
    ) -> Any:
        """Export roster in specified format"""
        if format == "dict":
            return [
                {
                    "wallet_id": c.wallet_id,
                    "address": c.address,
                    "classification": c.classification,
                    "quant_score": c.quant_score,
                    "copy_score": c.copy_score,
                    "performance_score": c.performance_score,
                    "risk_score": c.risk_score,
                    "copyability_score": c.copyability_score,
                    "top_rankings": c.top_rankings,
                    "key_metrics": c.key_metrics,
                    "recommendation": c.recommendation,
                }
                for c in roster
            ]
        elif format == "addresses":
            return [c.address for c in roster]
        else:
            raise ValueError(f"Unknown format: {format}")
