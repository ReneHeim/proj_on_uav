import numpy as np
import polars as pl
import pytest

from src.stats.Logistic_regression import (
    _parse_top_bin_key,
    format_logistic_results,
    logistic_regression,
    OLS,
    preprocess_healthy_diseased,
)


# ── preprocess_healthy_diseased ───────────────────────────────────────────────

class TestPreprocessHealthyDiseased:
    def test_returns_status_column(self):
        h = pl.DataFrame({
            "sunelev": [45, 60],
            "saa": [90, 180],
            "vaa": [90, 180],
            "vza": [10, 30],
            "band5": [0.5, 0.6],
        })
        d = pl.DataFrame({
            "sunelev": [30, 70],
            "saa": [270, 0],
            "vaa": [270, 0],
            "vza": [15, 50],
            "band5": [0.3, 0.4],
        })
        result = preprocess_healthy_diseased(h, d, sample_size=100)
        assert "status" in result.columns
        assert set(result["status"].unique()) == {"healthy", "diseased"}

    def test_has_expected_columns(self):
        h = pl.DataFrame({
            "sunelev": [45, 60],
            "saa": [90, 180],
            "vaa": [90, 180],
            "vza": [10, 30],
            "band5": [0.5, 0.6],
        })
        d = pl.DataFrame({
            "sunelev": [30, 70],
            "saa": [270, 0],
            "vaa": [270, 0],
            "vza": [15, 50],
            "band5": [0.3, 0.4],
        })
        result = preprocess_healthy_diseased(h, d, sample_size=100)
        for col in ["sza", "RAA", "raa_bin", "vza_bin"]:
            assert col in result.columns

    def test_sample_size_respected(self):
        h = pl.DataFrame({
            "sunelev": [45] * 200,
            "saa": [90] * 200,
            "vaa": [90] * 200,
            "vza": [10] * 200,
            "band5": np.random.uniform(0, 1, 200),
        })
        d = pl.DataFrame({
            "sunelev": [30] * 200,
            "saa": [270] * 200,
            "vaa": [270] * 200,
            "vza": [50] * 200,
            "band5": np.random.uniform(0, 1, 200),
        })
        result = preprocess_healthy_diseased(h, d, sample_size=50)
        assert len(result) <= 50

    def test_data_fewer_than_sample_size(self):
        h = pl.DataFrame({
            "sunelev": [45],
            "saa": [90],
            "vaa": [90],
            "vza": [10],
            "band5": [0.5],
        })
        d = pl.DataFrame({
            "sunelev": [30],
            "saa": [270],
            "vaa": [270],
            "vza": [50],
            "band5": [0.3],
        })
        result = preprocess_healthy_diseased(h, d, sample_size=10_000)
        assert len(result) > 0

    def test_status_is_categorical(self):
        h = pl.DataFrame({
            "sunelev": [45, 60],
            "saa": [90, 180],
            "vaa": [90, 180],
            "vza": [10, 30],
            "band5": [0.5, 0.6],
        })
        d = pl.DataFrame({
            "sunelev": [30, 70],
            "saa": [270, 0],
            "vaa": [270, 0],
            "vza": [15, 50],
            "band5": [0.3, 0.4],
        })
        result = preprocess_healthy_diseased(h, d, sample_size=100)
        assert str(result["status"].dtype) == "category"


# ── format_logistic_results ───────────────────────────────────────────────────

class TestFormatLogisticResults:
    def _make_mock_result(self):
        return {
            "AUROC_metrics": {
                "AUROC_nadir": 0.85,
                "AUROC_main": 0.90,
                "AUROC_full": 0.92,
                "AUROC_angle": 0.88,
                "Δ_full−nadir": 0.07,
                "Δ_main−nadir": 0.05,
                "Δ_full−main": 0.02,
                "ΔAUROC_geo−nadir": 0.03,
            },
            "Effect_size": {
                "Cohen_d_nadir": 1.2,
                "top_bins_by_|d|": {
                    "vza_bin=0-20_raa_bin=-90-0": 2.5,
                    "vza_bin=20-40_raa_bin=0-90": 1.8,
                },
            },
        }

    def test_wide_shape_returns_single_row(self):
        data = self._make_mock_result()
        result = format_logistic_results(data, shape="wide")
        assert result.height == 1

    def test_wide_shape_has_auroc_columns(self):
        data = self._make_mock_result()
        result = format_logistic_results(data, shape="wide")
        assert "AUROC_nadir" in result.columns
        assert "Cohen_d_nadir" in result.columns

    def test_long_shape_returns_multiple_rows(self):
        data = self._make_mock_result()
        result = format_logistic_results(data, shape="long")
        assert result.height > 1

    def test_long_shape_has_expected_sections(self):
        data = self._make_mock_result()
        result = format_logistic_results(data, shape="long")
        sections = result["section"].unique().to_list()
        assert "AUROC" in sections
        assert "EffectSize" in sections

    def test_empty_results(self):
        empty = {
            "AUROC_metrics": {},
            "Effect_size": {},
        }
        result_wide = format_logistic_results(empty, shape="wide")
        assert result_wide.height == 1
        result_long = format_logistic_results(empty, shape="long")
        assert result_long.height == 0

    def test_invalid_shape_raises(self):
        data = self._make_mock_result()
        with pytest.raises(ValueError, match="shape must be 'wide' or 'long'"):
            format_logistic_results(data, shape="invalid")


# ── _parse_top_bin_key ───────────────────────────────────────────────────────

class TestParseTopBinKey:
    def test_standard_key(self):
        key = "vza_bin=0-20_raa_bin=-90-0"
        vb, rb, original = _parse_top_bin_key(key)
        assert vb == "0-20"
        assert rb == "-90-0"
        assert original == key

    def test_tuple_input(self):
        key = ("10-30", "45-90")
        vb, rb, original = _parse_top_bin_key(key)
        assert vb == "10-30"
        assert rb == "45-90"
        assert original == "vza_bin=10-30_raa_bin=45-90"

    def test_malformed_key(self):
        key = "garbage_string"
        vb, rb, original = _parse_top_bin_key(key)
        assert isinstance(vb, str)
        assert isinstance(rb, str)
        assert original == key

    def test_integer_input(self):
        key = 42
        vb, rb, original = _parse_top_bin_key(key)
        assert vb == ""
        assert rb == ""
        assert original == "42"


# ── OLS ───────────────────────────────────────────────────────────────────────

class TestOLS:
    def test_basic_call(self):
        pytest.importorskip("statsmodels")
        h = pl.DataFrame({
            "sunelev": np.random.uniform(30, 70, 100),
            "saa": np.random.uniform(0, 359, 100),
            "vaa": np.random.uniform(0, 359, 100),
            "vza": np.random.uniform(0, 80, 100),
            "band5": np.random.uniform(0, 1, 100),
        })
        d = pl.DataFrame({
            "sunelev": np.random.uniform(30, 70, 100),
            "saa": np.random.uniform(0, 359, 100),
            "vaa": np.random.uniform(0, 359, 100),
            "vza": np.random.uniform(0, 80, 100),
            "band5": np.random.uniform(0, 1, 100) + 0.5,
        })
        df = preprocess_healthy_diseased(h, d, sample_size=200)
        OLS(df)


# ── logistic_regression ───────────────────────────────────────────────────────

class TestLogisticRegression:
    def test_basic_call(self):
        pytest.importorskip("sklearn")
        h = pl.DataFrame({
            "sunelev": np.random.uniform(30, 70, 100),
            "saa": np.random.uniform(0, 359, 100),
            "vaa": np.random.uniform(0, 359, 100),
            "vza": np.random.uniform(0, 80, 100),
            "band5": np.random.uniform(0, 1, 100),
        })
        d = pl.DataFrame({
            "sunelev": np.random.uniform(30, 70, 100),
            "saa": np.random.uniform(0, 359, 100),
            "vaa": np.random.uniform(0, 359, 100),
            "vza": np.random.uniform(0, 80, 100),
            "band5": np.random.uniform(0.5, 1.5, 100),
        })
        df = preprocess_healthy_diseased(h, d, sample_size=200)
        result = logistic_regression(df)
        assert isinstance(result, pl.DataFrame)
        assert result.height == 1
        assert "AUROC_metrics" in result.columns
        assert "Effect_size" in result.columns
