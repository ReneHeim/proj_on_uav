import numpy as np
import polars as pl
import pytest

import src.models.angular_support_sensitivity as sensitivity
from src.stats.fair_multiangular_significance import imputed_compact_arrays, permuted_targets
from src.stats.all_feature_angular_test import angular_contrasts
from src.models.angular_support_sensitivity import (
    evaluate,
    evaluate_nested_support,
    build_threshold_features,
    fold_contrast_arrays,
    paired_multiangular_effects,
    retained_cells,
)
from src.models.matched_angular_validation import common_cells, evaluate_week, zone_for_vza


def test_common_cells_requires_presence_in_every_plot():
    cells = pl.DataFrame(
        {
            "plot_id": ["plot_0", "plot_0", "plot_1", "plot_1"],
            "vza_cell": [0, 2, 0, 4],
            "raa_cell": [0, 0, 0, 0],
        }
    )

    result = common_cells(cells, ["plot_0", "plot_1"])

    assert result.to_dicts() == [{"vza_cell": 0, "raa_cell": 0}]


def test_zone_assignment_uses_fine_cell_center():
    assert zone_for_vza(0) == "0_15"
    assert zone_for_vza(14) == "15_25"
    assert zone_for_vza(24) == "25_35"
    assert zone_for_vza(34) == "35_45"
    assert zone_for_vza(44) == "45_60"
    assert zone_for_vza(60) is None


def test_relaxed_support_is_required_within_each_target_class():
    targets = pl.DataFrame(
        {
            "plot_id": ["p0", "p1", "p2", "p3"],
            "future_disease_wk8": [0, 0, 1, 1],
        }
    )
    cells = pl.DataFrame(
        {
            "plot_id": ["p0", "p1", "p0", "p1", "p2", "p3"],
            "vza_cell": [0, 0, 2, 2, 2, 2],
            "raa_cell": [0, 0, 0, 0, 0, 0],
        }
    )

    result = retained_cells(cells, targets, targets["plot_id"].to_list(), 0.5)

    assert result.to_dicts() == [{"vza_cell": 2, "raa_cell": 0}]


def test_common_support_includes_target_plot_with_no_cells():
    cells = pl.DataFrame(
        {
            "plot_id": ["plot_0", "plot_1"],
            "vza_cell": [0, 0],
            "raa_cell": [0, 0],
        }
    )

    with pytest.raises(RuntimeError, match="shared by every target plot"):
        common_cells(cells, ["plot_0", "plot_1", "plot_2"])


def test_relaxed_support_counts_zero_coverage_target_plots():
    targets = pl.DataFrame(
        {
            "plot_id": ["p0", "p1", "p2", "p3"],
            "future_disease_wk8": [0, 0, 1, 1],
        }
    )
    cells = pl.DataFrame(
        {
            "plot_id": ["p0", "p1", "p2"],
            "vza_cell": [0, 0, 0],
            "raa_cell": [0, 0, 0],
        }
    )

    result = retained_cells(cells, targets, targets["plot_id"].to_list(), 0.7)

    assert result.is_empty()


def test_off_nadir_only_support_skips_nadir_and_contrast_models():
    plot_ids = [f"plot_{index}" for index in range(8)]
    features = pl.DataFrame(
        {
            "plot_id": plot_ids,
            "week": [0] * 8,
            "future_disease_wk8": [0, 1] * 4,
            "n_pixels": [1000 + index for index in range(8)],
            "n_images": [10 + index for index in range(8)],
            "vza_mean": [25.0 + index * 0.1 for index in range(8)],
            "vza_std": [2.0] * 8,
            "vza_min": [20.0] * 8,
            "vza_max": [30.0] * 8,
            "raa_mean": [45.0] * 8,
            "raa_std": [10.0] * 8,
            "present_v24_r000": [1.0] * 8,
            "band5_v24_r000": [0.2 + index * 0.01 for index in range(8)],
        }
    )
    columns = {
        "geometry": ["n_pixels", "n_images", "vza_mean", "vza_std", "vza_min", "vza_max", "raa_mean", "raa_std"],
        "presence": ["present_v24_r000"],
        "nadir": [],
        "absolute": ["band5_v24_r000"],
        "contrast": [],
    }

    result = evaluate(features, columns, week=0, threshold=0.8, n_cells=1, missing_fraction=0.0)

    assert set(result["feature_set"].unique().to_list()) == {
        "G_geometry",
        "P_presence",
        "A_fine_absolute",
        "A_geometry_residual",
    }


def test_nested_support_selection_uses_training_plots_only(monkeypatch):
    plot_ids = [f"plot_{index}" for index in range(8)]
    targets = pl.DataFrame(
        {
            "plot_id": plot_ids,
            "future_disease_wk8": [0, 1] * 4,
        }
    )
    geometry = pl.DataFrame(
        {
            "plot_id": plot_ids,
            "n_pixels": [1000] * 8,
            "n_images": [10] * 8,
            "vza_mean": [20.0] * 8,
            "vza_std": [2.0] * 8,
            "vza_min": [0.0] * 8,
            "vza_max": [40.0] * 8,
            "raa_mean": [45.0] * 8,
            "raa_std": [10.0] * 8,
        }
    )
    observed_support_rosters = []

    def fake_build(week, cells, geometry, targets, threshold, support_plot_ids=None, class_balanced=True):
        observed_support_rosters.append(set(support_plot_ids))
        features = targets.join(geometry, on="plot_id").with_columns(
            pl.arange(0, targets.height, eager=True).cast(pl.Float64).alias("band5_v00_r000"),
            pl.lit(1.0).alias("present_v00_r000"),
        )
        columns = {
            "geometry": ["n_pixels", "n_images", "vza_mean", "vza_std", "vza_min", "vza_max", "raa_mean", "raa_std"],
            "presence": ["present_v00_r000"],
            "nadir": ["band5_v00_r000"],
            "absolute": ["band5_v00_r000"],
            "contrast": [],
        }
        return features, columns, 1, 0.0

    monkeypatch.setattr(sensitivity, "build_threshold_features", fake_build)

    result = evaluate_nested_support(0, pl.DataFrame(), geometry, targets, 0.8)

    assert not result.is_empty()
    assert len(observed_support_rosters) >= 2
    assert all(roster < set(plot_ids) for roster in observed_support_rosters)


def test_matched_evaluation_skips_missing_nadir_columns():
    plot_ids = [f"plot_{index}" for index in range(8)]
    features = pl.DataFrame(
        {
            "plot_id": plot_ids,
            "week": [0] * 8,
            "future_disease_wk8": [0, 1] * 4,
            "n_pixels": [1000 + index for index in range(8)],
            "n_images": [10 + index for index in range(8)],
            "vza_mean": [25.0] * 8,
            "vza_std": [2.0] * 8,
            "vza_min": [20.0] * 8,
            "vza_max": [30.0] * 8,
            "raa_mean": [45.0] * 8,
            "raa_std": [10.0] * 8,
            "band5_matched_15_25": [0.2 + index * 0.01 for index in range(8)],
            **{
                f"pixels_vza_{lo}_{hi}": [500] * 8
                for lo, hi in [(0, 15), (15, 25), (25, 35), (35, 45), (45, 60)]
            },
        }
    )

    folds, _ = evaluate_week(features)

    assert "N_matched_nadir" not in folds["feature_set"].unique().to_list()


def test_paired_multiangular_effect_uses_identical_folds():
    folds = pl.DataFrame(
        {
            "week": [5] * 6,
            "support_threshold": [0.7] * 6,
            "feature_set": ["N_geometry_residual"] * 3 + ["A_geometry_residual"] * 3,
            "fold": [0, 1, 2, 0, 1, 2],
            "AUROC": [0.4, 0.5, 0.6, 0.6, 0.6, 0.7],
        }
    )

    result = paired_multiangular_effects(folds)

    row = result.row(0, named=True)
    assert row["baseline"] == "N_geometry_residual"
    assert row["comparator"] == "A_geometry_residual"
    assert row["n_paired_folds"] == 3
    assert row["delta_AUROC_mean"] == pytest.approx(0.1333333333)
    assert row["folds_improved"] == 3


def test_contrasts_impute_nadir_from_training_fold_before_subtraction():
    features = pl.DataFrame(
        {
            "band5_v00_r000": [1.0, 3.0, None],
            "band5_v20_r000": [5.0, 7.0, 9.0],
        }
    )
    sources = {
        "band5_contrast_v20_r000": {
            "off_nadir": "band5_v20_r000",
            "nadir": ["band5_v00_r000"],
        }
    }

    train, test = fold_contrast_arrays(features, sources, [0, 1], [2])

    assert train[:, 0].tolist() == pytest.approx([4.0, 4.0])
    assert test[:, 0].tolist() == pytest.approx([7.0])


def test_permuted_targets_preserve_plot_roster_and_class_counts():
    targets = pl.DataFrame(
        {
            "plot_id": [f"plot_{index}" for index in range(8)],
            "future_disease_wk8": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )

    shuffled = permuted_targets(targets, __import__("numpy").random.default_rng(7))

    assert shuffled["plot_id"].to_list() == targets["plot_id"].to_list()
    assert shuffled["future_disease_wk8"].sum() == targets["future_disease_wk8"].sum()


def test_compact_multiangular_adds_only_five_contrasts_to_nadir():
    data = {f"band{band}_nadir": [float(band), float(band + 1), float(band + 2)] for band in range(1, 6)}
    data.update({f"band{band}_off": [float(band + 2), float(band + 3), float(band + 4)] for band in range(1, 6)})
    features = pl.DataFrame(data)

    train_nadir, test_nadir, train_multi, test_multi = imputed_compact_arrays(features, [0, 1], [2])

    assert train_nadir.shape == (2, 5)
    assert test_nadir.shape == (1, 5)
    assert train_multi.shape == (2, 10)
    assert test_multi.shape == (1, 10)
    assert (train_multi[:, 5:] == 2.0).all()


def test_compact_arrays_reject_training_fold_without_nadir():
    data = {f"band{band}_nadir": [None, None, 0.2] for band in range(1, 6)}
    data.update({f"band{band}_off": [0.3, 0.4, 0.5] for band in range(1, 6)})

    with pytest.raises(RuntimeError, match="no observed nadir"):
        imputed_compact_arrays(pl.DataFrame(data), [0, 1], [2])


def test_contrast_sources_exclude_nadir_cells():
    rows = []
    for plot_id in ["p0", "p1"]:
        for vza_cell, value in [(0, 0.1), (16, 0.2)]:
            rows.append(
                {
                    "plot_id": plot_id,
                    "vza_cell": vza_cell,
                    "raa_cell": 0,
                    **{f"band{band}": value for band in range(1, 6)},
                }
            )
    cells = pl.DataFrame(rows)
    geometry = pl.DataFrame(
        {
            "plot_id": ["p0", "p1"],
            "n_pixels": [1000, 1000],
            "n_images": [2, 2],
            "vza_mean": [10.0, 10.0],
            "vza_std": [2.0, 2.0],
            "vza_min": [0.0, 0.0],
            "vza_max": [20.0, 20.0],
            "raa_mean": [0.0, 0.0],
            "raa_std": [0.0, 0.0],
        }
    )
    targets = pl.DataFrame({"plot_id": ["p0", "p1"], "future_disease_wk8": [0, 1]})

    _, columns, _, _ = build_threshold_features(0, cells, geometry, targets, 1.0)

    off_nadir_sources = [spec["off_nadir"] for spec in columns["contrast_sources"].values()]
    assert off_nadir_sources
    assert all("_v16_" in column for column in off_nadir_sources)


def test_contrast_model_rejects_training_fold_without_observed_nadir():
    features = pl.DataFrame(
        {
            "band5_nadir_reference": [None, None, 0.2],
            "band5_v16_r000": [0.3, 0.4, 0.5],
        }
    )
    sources = {
        "band5_contrast_v16_r000": {
            "off_nadir": "band5_v16_r000",
            "nadir": ["band5_nadir_reference"],
        }
    }

    with pytest.raises(RuntimeError, match="no observed nadir"):
        fold_contrast_arrays(features, sources, [0, 1], [2])


def test_all_feature_transform_preserves_nadir_and_maps_each_band_contrast():
    nadir = [1.0, 2.0, 3.0, 4.0, 5.0]
    off = []
    for band_value in nadir:
        off.extend([band_value + 0.1, band_value + 0.2])
    transformed = angular_contrasts(np.asarray([nadir + off]))

    assert transformed.shape == (1, 15)
    assert transformed[0, :5].tolist() == nadir
    assert transformed[0, 5:].tolist() == pytest.approx([0.1, 0.2] * 5)
