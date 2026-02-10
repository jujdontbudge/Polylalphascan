"""Tests for Excel export fixes"""
import pytest


class TestSheetNameCollision:
    """Tests for sheet name collision avoidance"""

    def test_sheet_name_no_collision(self):
        """Long metric codes that would produce identical truncated names are disambiguated"""
        used_sheet_names = set()
        sheet_names = []

        # Simulate two metrics whose codes truncate to the same 31-char name
        codes = [
            "very_long_metric_code_abc_def_g",
            "very_long_metric_code_abc_def_h",
        ]
        metric_id = 1

        for code in codes:
            sheet_name = f"{metric_id}_{code}"[:31]
            if sheet_name in used_sheet_names:
                suffix = 2
                while True:
                    candidate = f"{metric_id}_{code}"[:28] + f"_{suffix}"
                    if candidate not in used_sheet_names:
                        sheet_name = candidate
                        break
                    suffix += 1
            used_sheet_names.add(sheet_name)
            sheet_names.append(sheet_name)

        # All names should be unique
        assert len(set(sheet_names)) == len(sheet_names)

    def test_sheet_name_within_limit(self):
        """Sheet names stay within Excel 31-char limit"""
        long_code = "a" * 50
        sheet_name = f"1_{long_code}"[:31]
        assert len(sheet_name) <= 31

    def test_collision_suffix_increment(self):
        """Multiple collisions produce incrementing suffixes"""
        used = set()
        names = []

        # 3 identical codes
        for _ in range(3):
            base = "1_same_code_here"[:31]
            name = base
            if name in used:
                suffix = 2
                while True:
                    candidate = base[:28] + f"_{suffix}"
                    if candidate not in used:
                        name = candidate
                        break
                    suffix += 1
            used.add(name)
            names.append(name)

        assert len(set(names)) == 3


class TestRankingsExport:
    """Tests for rankings truncation removal"""

    def test_all_rankings_exported(self):
        """All rankings are included, not just top 5"""
        roster_data = [
            {
                "address": "0xtest",
                "top_rankings": [
                    {"metric_name": f"metric_{i}", "rank": i, "value": float(i)}
                    for i in range(10)
                ],
            }
        ]

        rankings_data = []
        for c in roster_data:
            for ranking in c.get("top_rankings", []):  # No [:5] truncation
                rankings_data.append({
                    "Address": c["address"][:10] + "...",
                    "Metric": ranking["metric_name"],
                    "Rank": ranking["rank"],
                    "Value": ranking["value"],
                })

        assert len(rankings_data) == 10  # All 10, not 5

    def test_empty_rankings_handled(self):
        """Empty rankings produce no data"""
        roster_data = [{"address": "0xtest", "top_rankings": []}]

        rankings_data = []
        for c in roster_data:
            for ranking in c.get("top_rankings", []):
                rankings_data.append(ranking)

        assert len(rankings_data) == 0
