"""Tests for configuration"""
import pytest
import yaml
from pathlib import Path

from config.settings import (
    APIConfig,
    ScannerConfig,
    ClassifierConfig,
    MetricsConfig,
    RealtimeConfig,
    api_config,
    scanner_config,
    classifier_config,
    metrics_config,
    realtime_config,
)


class TestAPIConfig:
    """Tests for API configuration"""

    def test_base_urls(self):
        """Test API base URLs are configured"""
        assert api_config.GAMMA_API_BASE.startswith("https://")
        assert api_config.DATA_API_BASE.startswith("https://")
        assert api_config.CLOB_API_BASE.startswith("https://")

    def test_rate_limits(self):
        """Test rate limits are configured"""
        assert api_config.DATA_API_LIMIT > 0
        assert api_config.GAMMA_API_LIMIT > 0
        assert api_config.POSITIONS_API_LIMIT > 0

    def test_delays(self):
        """Test request delays are configured"""
        assert api_config.DATA_API_DELAY >= 0.1
        assert api_config.GAMMA_API_DELAY >= 0.1
        assert api_config.DEFAULT_DELAY >= 0.1


class TestScannerConfig:
    """Tests for scanner configuration"""

    def test_filter_thresholds(self):
        """Test lite filter thresholds"""
        assert scanner_config.MIN_VOLUME_USD == 500.0
        assert scanner_config.MIN_TRADE_COUNT == 10

    def test_ranking_settings(self):
        """Test ranking settings"""
        assert scanner_config.TOP_N == 100
        assert scanner_config.BUFFER_N == 150
        assert scanner_config.BUFFER_N > scanner_config.TOP_N

    def test_watering_hole_tags(self):
        """Test watering hole tags are configured"""
        tags = scanner_config.WATERING_HOLE_TAGS
        assert isinstance(tags, list)
        assert len(tags) > 0
        assert "crypto" in tags
        assert "politics" in tags
        assert "sports" in tags


class TestClassifierConfig:
    """Tests for classifier configuration"""

    def test_thresholds(self):
        """Test classification thresholds"""
        assert classifier_config.BOT_THRESHOLD == 80
        assert classifier_config.QUANT_THRESHOLD == 50
        assert classifier_config.HUMAN_THRESHOLD == 20

    def test_threshold_ordering(self):
        """Test thresholds are in correct order"""
        assert classifier_config.BOT_THRESHOLD > classifier_config.QUANT_THRESHOLD
        assert classifier_config.QUANT_THRESHOLD > classifier_config.HUMAN_THRESHOLD

    def test_labels(self):
        """Test classification labels"""
        assert classifier_config.LABEL_BOT == "bot"
        assert classifier_config.LABEL_QUANT == "quant"
        assert classifier_config.LABEL_HUMAN == "human"
        assert classifier_config.LABEL_UNKNOWN == "unknown"


class TestMetricsConfig:
    """Tests for metrics configuration"""

    def test_total_metrics(self):
        """Test total metrics count"""
        assert metrics_config.TOTAL_METRICS == 117
        assert metrics_config.TOTAL_GROUPS == 15

    def test_groups_structure(self):
        """Test groups dictionary structure"""
        groups = metrics_config.GROUPS
        assert len(groups) == 15

        for group_id in "ABCDEFGHIJKLMNO":
            assert group_id in groups
            group = groups[group_id]
            assert "name" in group
            assert "start" in group
            assert "end" in group
            assert "count" in group

    def test_group_ranges_contiguous(self):
        """Test metric ID ranges are contiguous"""
        groups = metrics_config.GROUPS
        sorted_groups = sorted(groups.items(), key=lambda x: x[1]["start"])

        expected_start = 1
        for group_id, group in sorted_groups:
            assert group["start"] == expected_start
            assert group["end"] == group["start"] + group["count"] - 1
            expected_start = group["end"] + 1

        # Should end at 117
        last_group = sorted_groups[-1][1]
        assert last_group["end"] == 117

    def test_group_counts(self):
        """Test group metric counts"""
        groups = metrics_config.GROUPS

        # Verify specific groups
        assert groups["A"]["count"] == 10  # Profitability
        assert groups["B"]["count"] == 11  # Risk & Stability
        assert groups["N"]["count"] == 5   # Bot/Quant Confirmation
        assert groups["O"]["count"] == 4   # Portfolio-Level

    def test_total_metrics_matches_sum(self):
        """Test total metrics equals sum of all groups"""
        total = sum(g["count"] for g in metrics_config.GROUPS.values())
        assert total == metrics_config.TOTAL_METRICS


class TestRealtimeConfig:
    """Tests for realtime capture configuration"""

    def test_websocket_config(self):
        """Test WebSocket configuration"""
        assert realtime_config.RTDS_WS_URL.startswith("wss://")
        assert realtime_config.WS_PING_INTERVAL > 0
        assert realtime_config.WS_RECONNECT_MAX > 0

    def test_rest_config(self):
        """Test REST backup configuration"""
        assert realtime_config.REST_POLL_INTERVAL > 0
        assert realtime_config.REST_OVERLAP_SECONDS > 0

    def test_checkpoint_config(self):
        """Test checkpoint configuration"""
        assert realtime_config.CHECKPOINT_INTERVAL > 0
        assert realtime_config.CHECKPOINT_OVERLAP > 0

    def test_deduplication_config(self):
        """Test deduplication configuration"""
        assert realtime_config.BLOOM_CAPACITY > 0
        assert 0 < realtime_config.BLOOM_ERROR_RATE < 1
        assert realtime_config.RECENT_CACHE_SIZE > 0


class TestMetricsYAML:
    """Tests for metrics_config.yaml"""

    @pytest.fixture
    def metrics_yaml(self):
        """Load metrics YAML file"""
        config_path = Path(__file__).parent.parent / "config" / "metrics_config.yaml"
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_yaml_has_117_metrics(self, metrics_yaml):
        """Test YAML contains 117 metrics"""
        metrics = metrics_yaml.get("metrics", [])
        assert len(metrics) == 117

    def test_metric_ids_are_unique(self, metrics_yaml):
        """Test all metric IDs are unique"""
        metrics = metrics_yaml.get("metrics", [])
        ids = [m["id"] for m in metrics]
        assert len(ids) == len(set(ids))

    def test_metric_ids_sequential(self, metrics_yaml):
        """Test metric IDs are 1-117"""
        metrics = metrics_yaml.get("metrics", [])
        ids = sorted([m["id"] for m in metrics])
        assert ids == list(range(1, 118))

    def test_metric_codes_unique(self, metrics_yaml):
        """Test all metric codes are unique"""
        metrics = metrics_yaml.get("metrics", [])
        codes = [m["code"] for m in metrics]
        assert len(codes) == len(set(codes))

    def test_metric_groups_valid(self, metrics_yaml):
        """Test all metrics have valid group IDs"""
        metrics = metrics_yaml.get("metrics", [])
        valid_groups = set("ABCDEFGHIJKLMNO")

        for metric in metrics:
            assert metric["group_id"] in valid_groups

    def test_metric_required_fields(self, metrics_yaml):
        """Test all metrics have required fields"""
        metrics = metrics_yaml.get("metrics", [])
        required = ["id", "group_id", "code", "name", "higher_is_better"]

        for metric in metrics:
            for field in required:
                assert field in metric, f"Metric {metric.get('id')} missing {field}"

    def test_higher_is_better_is_boolean(self, metrics_yaml):
        """Test higher_is_better field is boolean"""
        metrics = metrics_yaml.get("metrics", [])

        for metric in metrics:
            assert isinstance(metric["higher_is_better"], bool), \
                f"Metric {metric['id']} higher_is_better is not boolean"
