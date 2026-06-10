import numpy as np
import polars as pl
import pytest

from src.stats.ANOVA import ANOVA, ANOVA_optimized, ANOVA_preprocess

# ── ANOVA_preprocess ──────────────────────────────────────────────────────────


class TestANOVAPreprocess:
    def test_adds_bin_columns(self):
        df = pl.DataFrame(
            {
                "vza": [5, 25, 45, 65, 75],
                "RAA": [-350, -200, -50, 100, 200],
            }
        )
        result = ANOVA_preprocess(df)
        assert "raa_bin" in result.columns
        assert "vza_bin" in result.columns

    def test_bin_labels_correct(self):
        df = pl.DataFrame(
            {
                "vza": [5, 25, 45, 65, 75],
                "RAA": [-350, -200, -50, 100, 200],
            }
        )
        result = ANOVA_preprocess(df)
        assert result["vza_bin"].to_list() == [
            "0_to_20",
            "20_to_40",
            "40_to_60",
            "60_to_80",
            "60_to_80",
        ]
        assert result["raa_bin"].to_list() == [
            "-360_to_-270",
            "-270_to_-180",
            "-90_to_0",
            "90_to_180",
            "180_to_270",
        ]

    def test_drops_values_outside_edges(self):
        df = pl.DataFrame(
            {
                "vza": [100, 25],
                "RAA": [100, -200],
            }
        )
        result = ANOVA_preprocess(df)
        assert result.height == 1

    def test_custom_edge_values(self):
        df = pl.DataFrame(
            {
                "vza": [1, 5, 9],
                "RAA": [10, 20, 30],
            }
        )
        result = ANOVA_preprocess(df, vza_edges=[0, 3, 6, 9], raa_edges=[0, 15, 30])
        assert result["vza_bin"].to_list() == ["0_to_3", "3_to_6", "6_to_9"]
        assert result["raa_bin"].to_list() == ["0_to_15", "15_to_30", "15_to_30"]

    def test_all_values_in_single_bin(self):
        df = pl.DataFrame(
            {
                "vza": [10, 15, 18],
                "RAA": [5, 8, 12],
            }
        )
        result = ANOVA_preprocess(df, vza_edges=[0, 20], raa_edges=[0, 20])
        assert result.height == 3
        assert all(b == "0_to_20" for b in result["vza_bin"])
        assert all(b == "0_to_20" for b in result["raa_bin"])

    def test_empty_dataframe(self):
        df = pl.DataFrame({"vza": [], "RAA": []}, schema=[("vza", pl.Float64), ("RAA", pl.Float64)])
        result = ANOVA_preprocess(df)
        assert result.height == 0
        assert "vza_bin" in result.columns
        assert "raa_bin" in result.columns

    def test_last_bin_includes_upper_bound(self):
        df = pl.DataFrame(
            {
                "vza": [80, 79],
                "RAA": [360, 359],
            }
        )
        result = ANOVA_preprocess(df)
        assert result.height == 2
        assert all(b == "60_to_80" for b in result["vza_bin"])
        assert all(b == "270_to_360" for b in result["raa_bin"])


# ── ANOVA_optimized ───────────────────────────────────────────────────────────


class TestANOVAOptimized:
    def test_groups_with_different_means(self, rng):
        n = 100
        g1 = pl.DataFrame({"group": "A", "value": rng.normal(0, 1, n)})
        g2 = pl.DataFrame({"group": "B", "value": rng.normal(5, 1, n)})
        g3 = pl.DataFrame({"group": "C", "value": rng.normal(10, 1, n)})
        df = pl.concat([g1, g2, g3])
        result = ANOVA_optimized(df, "value", col="group")
        assert result["F_global"][0] > 10

    def test_p_values_small_for_different_groups(self, rng):
        n = 200
        g1 = pl.DataFrame({"group": "X", "value": rng.normal(0, 1, n)})
        g2 = pl.DataFrame({"group": "Y", "value": rng.normal(3, 1, n)})
        g3 = pl.DataFrame({"group": "Z", "value": rng.normal(7, 1, n)})
        df = pl.concat([g1, g2, g3])
        result = ANOVA_optimized(df, "value", col="group")
        assert result["reject"].all()

    def test_reports_cohens_d(self):
        df = pl.DataFrame(
            {
                "grp": ["A"] * 50 + ["B"] * 50,
                "val": [1.0] * 50 + [3.0] * 50,
            }
        )
        result = ANOVA_optimized(df, "val", col="grp")
        assert "cohens_d" in result.columns
        assert result["cohens_d"][0] > 0.5

    def test_reject_is_boolean(self):
        df = pl.DataFrame(
            {
                "grp": ["A"] * 50 + ["B"] * 50,
                "val": [1.0] * 50 + [3.0] * 50,
            }
        )
        result = ANOVA_optimized(df, "val", col="grp")
        assert result["reject"].dtype == pl.Boolean

    def test_identical_distributions(self, rng):
        n = 100
        g1 = pl.DataFrame({"group": "A", "value": rng.normal(0, 1, n)})
        g2 = pl.DataFrame({"group": "B", "value": rng.normal(0, 1, n)})
        g3 = pl.DataFrame({"group": "C", "value": rng.normal(0, 1, n)})
        df = pl.concat([g1, g2, g3])
        result = ANOVA_optimized(df, "value", col="group")
        assert result["F_global"][0] < 5

    def test_very_small_sample_sizes(self):
        df = pl.DataFrame(
            {
                "grp": ["A", "A", "B", "B", "C", "C"],
                "val": [1.0, 1.5, 10.0, 10.5, 5.0, 5.5],
            }
        )
        result = ANOVA_optimized(df, "val", col="grp")
        assert result["F_global"][0] > 0
        assert not result.is_empty()

    def test_returns_expected_columns(self):
        df = pl.DataFrame(
            {
                "grp": ["A"] * 30 + ["B"] * 30,
                "val": [1.0] * 30 + [3.0] * 30,
            }
        )
        result = ANOVA_optimized(df, "val", col="grp")
        expected_cols = {
            "group1",
            "group2",
            "mean group 2 - group1",
            "CI+-",
            "p_text",
            "-log10_p",
            "cohens_d",
            "reject",
            "F_global",
            "degrees of freedom",
            "eta_sq",
        }
        assert expected_cols.issubset(set(result.columns))


# ── ANOVA (statsmodels version) ───────────────────────────────────────────────


class TestANOVA:
    def test_basic_functionality(self):
        sm = pytest.importorskip("statsmodels")
        df = pl.DataFrame(
            {
                "vza": [5, 5, 5, 25, 25, 25, 45, 45, 45],
                "RAA": [10, 10, 10, 30, 30, 30, 50, 50, 50],
                "band": [1.0, 1.2, 0.9, 2.5, 2.3, 2.7, 5.0, 4.8, 5.2],
            }
        )
        df = ANOVA_preprocess(df)
        result = ANOVA(df, "band")
        assert "F_global" in result.columns
        assert "cohens_d" in result.columns
        assert "reject" in result.columns
        assert result["F_global"][0] > 0
