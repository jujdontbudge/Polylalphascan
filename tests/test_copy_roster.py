"""Tests for copy roster normalization"""
import pytest
from unittest.mock import patch, MagicMock

from src.core.classifier.copy_roster import CopyRosterGenerator


class TestCopyRosterNormalization:
    """Tests for _calculate_component_score normalization fix"""

    def setup_method(self):
        with patch.object(CopyRosterGenerator, '__init__', lambda self: None):
            self.generator = CopyRosterGenerator()

    def test_normalization_sharpe_ratio(self):
        """Sharpe of 2.0 normalizes to ~57 (not clipped to 2)"""
        metrics = {11: 2.0}  # Sharpe ratio
        metric_weights = {11: ("sharpe_ratio", 0.3)}

        score = self.generator._calculate_component_score(metrics, metric_weights)

        # Sharpe range is (-2, 5), so 2.0 -> (2.0 - (-2)) / (5 - (-2)) * 100 = 57.14
        assert 55 < score < 60

    def test_normalization_profit_factor(self):
        """Profit factor of 3.0 normalizes correctly"""
        metrics = {4: 3.0}
        metric_weights = {4: ("profit_factor", 0.2)}

        score = self.generator._calculate_component_score(metrics, metric_weights)

        # Range (0, 10), so 3.0 -> 30.0
        assert score == pytest.approx(30.0, abs=1.0)

    def test_normalization_drawdown_lower_is_better(self):
        """Drawdown of 50% normalizes correctly as 'lower is better'"""
        metrics = {14: 50.0}
        metric_weights = {14: ("max_drawdown", -0.4)}

        score = self.generator._calculate_component_score(metrics, metric_weights)

        # Range (0, 100), 50% -> normalized = 50, inverted (weight < 0) -> 50
        assert score == pytest.approx(50.0, abs=1.0)

    def test_normalization_low_drawdown_scores_high(self):
        """Low drawdown (good) scores higher than high drawdown"""
        low_dd = {14: 10.0}
        high_dd = {14: 80.0}
        weights = {14: ("max_drawdown", -0.4)}

        low_score = self.generator._calculate_component_score(low_dd, weights)
        high_score = self.generator._calculate_component_score(high_dd, weights)

        assert low_score > high_score

    def test_normalization_win_rate_100_percent(self):
        """Win rate of 100% normalizes to 100"""
        metrics = {22: 100.0}
        metric_weights = {22: ("win_rate", 0.2)}

        score = self.generator._calculate_component_score(metrics, metric_weights)
        assert score == pytest.approx(100.0, abs=1.0)

    def test_normalization_roi_200_percent(self):
        """ROI at boundary (200%) normalizes to 100"""
        metrics = {2: 200.0}
        metric_weights = {2: ("total_roi", 0.3)}

        score = self.generator._calculate_component_score(metrics, metric_weights)
        assert score == pytest.approx(100.0, abs=1.0)

    def test_missing_metrics_handled(self):
        """Missing metrics don't crash, score is neutral default"""
        metrics = {}  # No metrics at all
        metric_weights = {
            2: ("total_roi", 0.3),
            11: ("sharpe_ratio", 0.3),
        }

        score = self.generator._calculate_component_score(metrics, metric_weights)
        assert score == 50.0  # Neutral default

    def test_partial_missing_metrics(self):
        """Some metrics present, some missing -> uses available ones"""
        metrics = {22: 80.0}  # Only win rate
        metric_weights = {
            22: ("win_rate", 0.2),
            11: ("sharpe_ratio", 0.3),  # Missing
        }

        score = self.generator._calculate_component_score(metrics, metric_weights)
        # Should use win rate (80/100 * 100 = 80) as the only component
        assert score == pytest.approx(80.0, abs=1.0)

    def test_none_metric_value_skipped(self):
        """Metric with None value is treated as missing"""
        metrics = {2: None, 22: 60.0}
        metric_weights = {
            2: ("total_roi", 0.3),
            22: ("win_rate", 0.2),
        }

        score = self.generator._calculate_component_score(metrics, metric_weights)
        # Only win rate used: 60/100 * 100 = 60
        assert score == pytest.approx(60.0, abs=1.0)
