import subprocess
from types import SimpleNamespace

import polars as pl
import pytest

import run_year_analysis
from src.models import compare_feature_sets
from src.models.early_warning_prediction import build_temporal_dataset
from src.models.feature_selection import assert_reflectance_only, reflectance_feature_columns
from src.models.future_severity_prediction import TARGET_COL, build_future_targets


def test_reflectance_feature_selector_excludes_metadata_and_geometry():
    cols = [
        "plot_id",
        "week",
        "year",
        "cult",
        "trt",
        "disease_label",
        "cx",
        "cy",
        "vza",
        "band5_vza25_35",
        "band4_vza45_60_raa90_180",
        "band3_vza_diff_25_35",
        "ndvi_nadir",
    ]

    assert reflectance_feature_columns(cols) == [
        "band5_vza25_35",
        "band4_vza45_60_raa90_180",
        "band3_vza_diff_25_35",
        "ndvi_nadir",
    ]


def test_reflectance_feature_assertion_rejects_non_reflectance_predictors():
    with pytest.raises(ValueError, match="Non-reflectance predictors"):
        assert_reflectance_only(["band5_vza25_35", "year", "cx"], "unit-test")


def test_future_targets_use_observed_disease_label_not_treatment():
    metadata = pl.DataFrame(
        {
            "plot_id": ["plot_1", "plot_2"],
            "week": [8, 8],
            "year": [2024, 2024],
            "trt": ["trt", "no_trt"],
            "disease_label": [1, 0],
        }
    )

    targets = build_future_targets(metadata).sort("plot_id")

    assert targets[TARGET_COL].to_list() == [1, 0]


def test_future_targets_fail_without_observed_labels():
    metadata = pl.DataFrame(
        {
            "plot_id": ["plot_1", "plot_2"],
            "week": [8, 8],
            "year": [2024, 2024],
            "trt": ["trt", "no_trt"],
            "disease_label": [None, None],
        }
    )

    with pytest.raises(RuntimeError, match="refusing to derive labels from treatment"):
        build_future_targets(metadata)


def test_temporal_dataset_falls_back_to_plot_join_without_year():
    df = pl.DataFrame(
        {
            "plot_id": ["plot_1", "plot_1"],
            "week": [0, 1],
            "disease_label": [0, 1],
            "band1_mean": [0.2, 0.4],
        }
    )

    merged = build_temporal_dataset(df, 0, 1)

    assert merged.height == 1
    assert "year" not in merged.columns
    assert merged["target_label"].to_list() == [1]


def test_temporal_dataset_joins_on_year_when_available():
    df = pl.DataFrame(
        {
            "year": [2024, 2024, 2025, 2025],
            "plot_id": ["plot_1", "plot_1", "plot_1", "plot_1"],
            "week": [0, 1, 0, 1],
            "disease_label": [0, 1, 0, 0],
            "band1_mean": [0.2, 0.4, 0.3, 0.5],
        }
    )

    merged = build_temporal_dataset(df, 0, 1).sort("year")

    assert merged.height == 2
    assert merged["year"].to_list() == [2024, 2025]
    assert merged["target_label"].to_list() == [1, 0]


def test_compare_feature_sets_fails_when_requested_year_has_no_results(monkeypatch):
    monkeypatch.setattr(compare_feature_sets, "setup_logging", lambda: None)
    monkeypatch.setattr(compare_feature_sets, "FEATURE_SETS", ["M1"])
    monkeypatch.setattr(compare_feature_sets, "evaluate_feature_set", lambda name, year: [])

    with pytest.raises(RuntimeError, match="2025"):
        compare_feature_sets.main("2025")


def test_run_year_uses_year_specific_outputs_instead_of_stale_global_results(
    tmp_path, monkeypatch
):
    stale_global = tmp_path / "outputs" / "results"
    stale_global.mkdir(parents=True)
    (stale_global / "model_comparison_summary.csv").write_text("stale\n")
    (stale_global / "model_comparison_by_fold.csv").write_text("stale\n")

    year_results = tmp_path / "outputs" / "2025" / "results"
    year_results.mkdir(parents=True)
    (year_results / "old.csv").write_text("old\n")

    dummy_builder = SimpleNamespace(WEEK_DIRS={"2025_week0": tmp_path}, main=lambda: None)
    monkeypatch.setattr(run_year_analysis, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(run_year_analysis, "load_feature_builder", lambda: dummy_builder)

    def fake_run(*args, **kwargs):
        (year_results / "model_comparison_summary.csv").write_text("fresh-summary\n")
        (year_results / "model_comparison_by_fold.csv").write_text("fresh-folds\n")
        return subprocess.CompletedProcess(args[0], 0, stdout="AUROC\n", stderr="")

    monkeypatch.setattr(run_year_analysis.subprocess, "run", fake_run)

    run_year_analysis.run_year("2025")

    assert not (year_results / "old.csv").exists()
    assert (year_results / "model_comparison_summary.csv").read_text() == "fresh-summary\n"
    assert (year_results / "model_comparison_by_fold.csv").read_text() == "fresh-folds\n"


def test_run_year_fails_if_subprocess_writes_no_year_results(tmp_path, monkeypatch):
    dummy_builder = SimpleNamespace(WEEK_DIRS={"2025_week0": tmp_path}, main=lambda: None)
    monkeypatch.setattr(run_year_analysis, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(run_year_analysis, "load_feature_builder", lambda: dummy_builder)
    monkeypatch.setattr(
        run_year_analysis.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout="", stderr=""),
    )

    with pytest.raises(RuntimeError, match="no usable result files"):
        run_year_analysis.run_year("2025")
