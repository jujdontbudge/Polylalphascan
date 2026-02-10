"""
Bot/Quant Detector - Classifies wallets based on trading patterns.

Uses Group N metrics (Bot/Quant Confirmation) as primary signals:
- 109: Trade Interval Entropy (low = bot)
- 110: Round Number Avoidance (high = bot)
- 111: Latency Consistency (low variance = bot)
- 112: Multi-Market Simultaneity (human can't do this)
- 113: Session Pattern Regularity (clock-like = cron job)

Classification:
- Quant Score >= 80 → "bot"
- Quant Score >= 50 → "quant"
- Quant Score <= 20 → "human"
"""
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from src.database.repositories import WalletRepository, MetricRepository
from src.database.models import get_session, Wallet
from config.settings import classifier_config

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of wallet classification"""
    wallet_id: int
    address: str
    classification: str
    quant_score: float
    confidence: float  # 0-1
    signals: Dict[str, Any]
    recommendation: str


class BotDetector:
    """
    Detects algorithmic traders (bots/quants) from trading patterns.

    Uses a multi-factor approach combining:
    1. Group N metrics (primary bot signals)
    2. Trading consistency patterns
    3. Execution quality indicators
    """

    # Metric IDs for bot detection (Group N)
    BOT_METRIC_IDS = [109, 110, 111, 112, 113]

    # Additional metrics that indicate bot behavior
    SUPPORTING_METRIC_IDS = {
        66: "strategy_consistency",  # High consistency = bot
        74: "discipline_score",       # High discipline = bot
        107: "execution_speed",       # Fast execution = bot
    }

    def __init__(self):
        self.bot_threshold = classifier_config.BOT_THRESHOLD
        self.quant_threshold = classifier_config.QUANT_THRESHOLD
        self.human_threshold = classifier_config.HUMAN_THRESHOLD

    def classify_wallet(
        self,
        wallet_id: int,
        metrics: Dict[int, float] = None,
    ) -> ClassificationResult:
        """
        Classify a single wallet.

        Args:
            wallet_id: Wallet database ID
            metrics: Optional pre-fetched metrics

        Returns:
            ClassificationResult with classification details
        """
        session = get_session()

        try:
            with WalletRepository(session) as wallet_repo, \
                 MetricRepository(session) as metric_repo:

                wallet = wallet_repo.get_by_id(wallet_id)
                if not wallet:
                    return self._create_error_result(wallet_id, "Wallet not found")

                # Fetch metrics if not provided
                if metrics is None:
                    metrics = metric_repo.get_wallet_metrics(wallet_id)

                # Calculate quant score from bot metrics
                quant_score = self._calculate_quant_score(metrics)

                # Determine classification
                classification = self._classify_from_score(quant_score)

                # Calculate confidence
                confidence = self._calculate_confidence(metrics, quant_score)

                # Extract signals
                signals = self._extract_signals(metrics)

                # Generate recommendation
                recommendation = self._generate_recommendation(
                    classification, quant_score, signals
                )

                # Update wallet classification
                wallet_repo.set_classification(
                    wallet_id, classification, quant_score
                )

                return ClassificationResult(
                    wallet_id=wallet_id,
                    address=wallet.address,
                    classification=classification,
                    quant_score=quant_score,
                    confidence=confidence,
                    signals=signals,
                    recommendation=recommendation,
                )

        finally:
            session.close()

    def _calculate_quant_score(self, metrics: Dict[int, float]) -> float:
        """Calculate quant score from bot detection metrics"""
        # Weights for each metric (same as in group_n_bot_quant.py)
        weights = {
            109: -0.2,  # Entropy (lower = more bot, so negative weight)
            110: 0.25,  # Round number avoidance (higher = more bot)
            111: -0.2,  # Latency consistency (lower = more bot)
            112: 0.2,   # Simultaneity (higher = more bot)
            113: 0.15,  # Session regularity (higher = more bot)
        }

        score = 50  # Start at neutral

        for metric_id, weight in weights.items():
            if metric_id in metrics and metrics[metric_id] is not None:
                value = metrics[metric_id]

                if weight < 0:
                    contribution = (100 - value) * abs(weight)
                else:
                    contribution = value * weight

                score += contribution - 50 * abs(weight)

        return max(0, min(100, score))

    def _classify_from_score(self, quant_score: float) -> str:
        """Determine classification from quant score"""
        if quant_score >= self.bot_threshold:
            return classifier_config.LABEL_BOT
        elif quant_score >= self.quant_threshold:
            return classifier_config.LABEL_QUANT
        elif quant_score <= self.human_threshold:
            return classifier_config.LABEL_HUMAN
        else:
            return classifier_config.LABEL_UNKNOWN

    def _calculate_confidence(
        self,
        metrics: Dict[int, float],
        quant_score: float,
    ) -> float:
        """Calculate confidence level of classification"""
        # Base confidence from how extreme the quant score is
        if quant_score >= 80 or quant_score <= 20:
            base_confidence = 0.9
        elif quant_score >= 60 or quant_score <= 40:
            base_confidence = 0.7
        else:
            base_confidence = 0.5

        # Adjust based on available metrics
        available_bot_metrics = sum(
            1 for mid in self.BOT_METRIC_IDS
            if mid in metrics and metrics[mid] is not None
        )
        data_factor = available_bot_metrics / len(self.BOT_METRIC_IDS)

        return base_confidence * data_factor

    def _extract_signals(self, metrics: Dict[int, float]) -> Dict[str, Any]:
        """Extract readable signals from metrics"""
        signals = {}

        # Bot detection signals
        if 109 in metrics and metrics[109] is not None:
            entropy = metrics[109]
            signals["interval_pattern"] = {
                "value": entropy,
                "interpretation": "Regular" if entropy < 40 else "Random" if entropy > 60 else "Mixed",
                "bot_indicator": entropy < 40,
            }

        if 110 in metrics and metrics[110] is not None:
            round_avoid = metrics[110]
            signals["price_precision"] = {
                "value": round_avoid,
                "interpretation": "Avoids round numbers" if round_avoid > 60 else "Uses round numbers",
                "bot_indicator": round_avoid > 60,
            }

        if 111 in metrics and metrics[111] is not None:
            latency = metrics[111]
            signals["timing_consistency"] = {
                "value": latency,
                "interpretation": "Consistent" if latency < 40 else "Variable",
                "bot_indicator": latency < 40,
            }

        if 112 in metrics and metrics[112] is not None:
            simultaneity = metrics[112]
            signals["multi_market"] = {
                "value": simultaneity,
                "interpretation": f"{simultaneity:.0f} simultaneous trades detected",
                "bot_indicator": simultaneity > 5,
            }

        if 113 in metrics and metrics[113] is not None:
            regularity = metrics[113]
            signals["session_pattern"] = {
                "value": regularity,
                "interpretation": "Clock-like" if regularity > 70 else "Irregular",
                "bot_indicator": regularity > 70,
            }

        return signals

    def _generate_recommendation(
        self,
        classification: str,
        quant_score: float,
        signals: Dict[str, Any],
    ) -> str:
        """Generate human-readable recommendation"""
        bot_indicators = sum(
            1 for s in signals.values()
            if isinstance(s, dict) and s.get("bot_indicator", False)
        )

        if classification == "bot":
            return (
                f"HIGH CONFIDENCE BOT (score: {quant_score:.0f}). "
                f"{bot_indicators}/{len(signals)} bot indicators triggered. "
                "Highly suitable for copy-trading due to consistent, algorithmic behavior."
            )
        elif classification == "quant":
            return (
                f"LIKELY QUANT/SEMI-AUTOMATED (score: {quant_score:.0f}). "
                f"{bot_indicators}/{len(signals)} bot indicators triggered. "
                "Shows systematic patterns but may have manual elements. "
                "Consider for copy-trading with monitoring."
            )
        elif classification == "human":
            return (
                f"LIKELY HUMAN (score: {quant_score:.0f}). "
                f"{bot_indicators}/{len(signals)} bot indicators triggered. "
                "Trading patterns suggest manual decision-making. "
                "Less suitable for copy-trading due to unpredictable timing."
            )
        else:
            return (
                f"UNKNOWN (score: {quant_score:.0f}). "
                "Insufficient data or ambiguous patterns. "
                "Gather more trade data before classification."
            )

    def _create_error_result(
        self,
        wallet_id: int,
        error: str,
    ) -> ClassificationResult:
        """Create error result"""
        return ClassificationResult(
            wallet_id=wallet_id,
            address="",
            classification="unknown",
            quant_score=0.0,
            confidence=0.0,
            signals={"error": error},
            recommendation=f"Error: {error}",
        )

    def batch_classify(
        self,
        wallet_ids: List[int],
    ) -> List[ClassificationResult]:
        """Classify multiple wallets"""
        return [self.classify_wallet(wid) for wid in wallet_ids]

    def get_bots(self, limit: int = 100) -> List[Wallet]:
        """Get wallets classified as bots"""
        session = get_session()
        try:
            with WalletRepository(session) as repo:
                return repo.get_by_classification("bot", limit=limit)
        finally:
            session.close()

    def get_quants(self, limit: int = 100) -> List[Wallet]:
        """Get wallets classified as quants"""
        session = get_session()
        try:
            with WalletRepository(session) as repo:
                return repo.get_by_classification("quant", limit=limit)
        finally:
            session.close()

    def get_classification_stats(self) -> Dict[str, int]:
        """Get classification distribution"""
        session = get_session()
        try:
            with WalletRepository(session) as repo:
                return repo.count_by_classification()
        finally:
            session.close()
