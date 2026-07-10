#!/usr/bin/env python3
"""Current severity from curve-only functional reflectance inputs.

This is the clean test of the claim that the angular reflectance curve itself
can predict plot severity.  The model receives sampled reflectance functions
over VZA, optionally stratified by broad reliable RAA bins.  It does not use
compact hand-engineered multiangular summaries, residual correction from another
model, treatment, cultivar, block, inoculation metadata, or disease history.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[5]
MPLCONFIG_DIR = ROOT / "outputs/.matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.severity import (
    analyze_current_plot_severity_2024_to_2025 as current_severity,
)
from scripts.analysis.severity import (
    debug_multiangular_rmse_bottleneck as residual_pipeline,
)
from scripts.analysis.severity.experiments.curve_shape.analyze_current_severity_magnitude_shape_functional_2024_to_2025 import (
    COVARIATES,
    META_COLS,
    MIN_GROUP_ANGLES,
    load_context_predictions,
    paired_delta_vs_nadir,
    pivot_curves,
    read_inputs,
    score_prediction_frame,
    shifted_log_values,
)
from scripts.analysis.severity.experiments.curve_shape.analyze_current_severity_sparse_functional_discriminant_shape_2024_to_2025 import (
    RAA_2024,
    RAA_2025,
    pivot_raa_vza_curves,
    read_raa,
)
from scripts.analysis.severity.analyze_multiangular_distribution_feature_family import (
    clean_token,
    markdown_table,
)

OUTPUT_ROOT = ROOT / "outputs/runs/analysis/severity/current/curve_only_functional_2024_to_2025"
RESULTS_DIR = OUTPUT_ROOT / "results"
REPORTS_DIR = OUTPUT_ROOT / "reports"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = OUTPUT_ROOT / "figures"
LOGS_DIR = ROOT / "outputs/archive/legacy_unscoped/logs"

TARGET = current_severity.TARGET
SEED = current_severity.SEED
EPS = 1e-4


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = (
        LOGS_DIR / f"analyze_current_severity_curve_only_functional_2024_to_2025_{timestamp}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        force=True,
    )
    logging.info("Log file: %s", log_path)
    return log_path


def log_phase(name: str, started: float) -> None:
    logging.info("[PHASE] %s: %.1fs", name, time.perf_counter() - started)


def configure_reused_pipeline_paths() -> None:
    residual_pipeline.ROOT = ROOT
    residual_pipeline.COVARIATES = COVARIATES
    residual_pipeline.OUTPUT_ROOT = OUTPUT_ROOT
    residual_pipeline.RESULTS_DIR = RESULTS_DIR
    residual_pipeline.REPORTS_DIR = REPORTS_DIR
    residual_pipeline.FIGURES_DIR = FIGURES_DIR
    residual_pipeline.PREDICTIONS_DIR = PREDICTIONS_DIR
    residual_pipeline.FROZEN_MANIFEST_PATH = OUTPUT_ROOT / "curve_only_functional_manifest.json"


def angle_columns(
    train_group: pd.DataFrame, test_group: pd.DataFrame
) -> tuple[list[str], np.ndarray]:
    train_angles = {
        float(col.replace("angle_", ""))
        for col in train_group.columns
        if isinstance(col, str) and col.startswith("angle_")
    }
    test_angles = {
        float(col.replace("angle_", ""))
        for col in test_group.columns
        if isinstance(col, str) and col.startswith("angle_")
    }
    angles = np.asarray(sorted(train_angles.intersection(test_angles)), dtype=float)
    cols = [f"angle_{angle:04.1f}" for angle in angles]
    return cols, angles


def merge_feature_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=META_COLS)
    merged = frames[0]
    for frame in frames[1:]:
        feature_cols = [col for col in frame.columns if col not in META_COLS]
        merged = merged.merge(frame[META_COLS + feature_cols], on=META_COLS, how="outer")
    return merged


def sampled_curve_features_for_group(
    train_group: pd.DataFrame,
    test_group: pd.DataFrame,
    source: str,
    group: str,
    cols: list[str],
    angles: np.ndarray,
    transforms: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    imputer = SimpleImputer(strategy="median")
    train_values = imputer.fit_transform(train_group[cols])
    test_values = imputer.transform(test_group[cols])
    train_log, test_log, shift = shifted_log_values(train_values, test_values)
    arrays = {
        "log": (train_log, test_log),
        "shape": (train_log - train_log[:, [0]], test_log - test_log[:, [0]]),
    }
    prefix = f"curveonly__{clean_token(source)}__{clean_token(group)}"
    train_out = train_group[META_COLS].reset_index(drop=True).copy()
    test_out = test_group[META_COLS].reset_index(drop=True).copy()
    for transform in transforms:
        train_arr, test_arr = arrays[transform]
        for idx, angle in enumerate(angles):
            name = f"{prefix}__{transform}__vza_{angle:04.1f}"
            train_out[name] = train_arr[:, idx]
            test_out[name] = test_arr[:, idx]
    audit = {
        "source": source,
        "curve_group": group,
        "n_train_curves": len(train_group),
        "n_test_curves": len(test_group),
        "n_angles": len(cols),
        "transforms": "+".join(transforms),
        "n_features": len(angles) * len(transforms),
        "shift_for_log": shift,
    }
    return train_out, test_out, audit


def build_sampled_curve_features(
    train_sources: dict[str, pd.DataFrame],
    test_sources: dict[str, pd.DataFrame],
    selected_sources: list[str],
    transforms: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    started = time.perf_counter()
    train_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for source in selected_sources:
        train_pivot = train_sources[source]
        test_pivot = test_sources[source]
        common_groups = sorted(
            set(train_pivot["curve_group"]).intersection(test_pivot["curve_group"])
        )
        for group in common_groups:
            train_group = train_pivot[train_pivot["curve_group"].eq(group)].copy()
            test_group = test_pivot[test_pivot["curve_group"].eq(group)].copy()
            cols, angles = angle_columns(train_group, test_group)
            if len(cols) < MIN_GROUP_ANGLES:
                continue
            train_out, test_out, audit = sampled_curve_features_for_group(
                train_group, test_group, source, group, cols, angles, transforms
            )
            train_frames.append(train_out)
            test_frames.append(test_out)
            audits.append(audit)
    train = merge_feature_frames(train_frames)
    test = merge_feature_frames(test_frames)
    audit = pd.DataFrame(audits)
    logging.info(
        "%s/%s curve-only features: train=%d rows/%d features test=%d rows/%d features",
        "+".join(selected_sources),
        "+".join(transforms),
        len(train),
        train.shape[1] - len(META_COLS),
        len(test),
        test.shape[1] - len(META_COLS),
    )
    log_phase(f"build curve-only {'+'.join(selected_sources)} {'+'.join(transforms)}", started)
    return train, test, audit


def make_curve_sources(
    long_2024: pd.DataFrame,
    long_2025: pd.DataFrame,
    raa_2024: pd.DataFrame,
    raa_2025: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    train = {
        "vza": pivot_curves(long_2024),
        "vza_raa_reliable": pivot_raa_vza_curves(raa_2024, reliable_only=True),
    }
    test = {
        "vza": pivot_curves(long_2025),
        "vza_raa_reliable": pivot_raa_vza_curves(raa_2025, reliable_only=True),
    }
    for source in train:
        logging.info(
            "%s source: train=%d rows/%d groups test=%d rows/%d groups",
            source,
            len(train[source]),
            train[source]["curve_group"].nunique(),
            len(test[source]),
            test[source]["curve_group"].nunique(),
        )
    return train, test


def evaluate_feature_set(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    disease_2024: pd.DataFrame,
    disease_2025: pd.DataFrame,
    feature_set: str,
) -> tuple[list[dict[str, object]], dict[tuple[str, str], pd.DataFrame], pd.DataFrame]:
    train = current_severity.build_current_model_table(train_features, disease_2024)
    test = current_severity.build_current_model_table(test_features, disease_2025)
    results: list[dict[str, object]] = []
    predictions: dict[tuple[str, str], pd.DataFrame] = {}
    selections: list[pd.DataFrame] = []
    for top_k in [20, 50]:
        result, pred, selection = current_severity.current_hurdle_stability_topk_model(
            train,
            test,
            feature_set,
            top_k=top_k,
            log_positive=False,
        )
        result["source"] = "curve_only_functional"
        results.append(result)
        predictions[(result["model"], feature_set)] = pred
        selections.append(selection)
    return results, predictions, pd.concat(selections, ignore_index=True)


def write_report(
    comparison: pd.DataFrame,
    delta: pd.DataFrame,
    audit: pd.DataFrame,
    selection: pd.DataFrame,
    paths: dict[str, Path],
    log_path: Path,
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "curve_only_functional_current_severity_summary.md"
    display_cols = [
        "model",
        "feature_set",
        "source",
        "n_test",
        "n_features",
        "rmse",
        "mae",
        "r2",
        "spearman",
        "train_in_sample_rmse",
        "train_grouped_oof_rmse",
        "external_minus_oof_rmse",
    ]
    display = comparison.copy()
    for col in display_cols:
        if col not in display.columns:
            display[col] = math.nan
    selected_summary = (
        selection[selection["selected_for_final_model"]]
        .groupby(["model", "feature_set", "role"], dropna=False)
        .size()
        .reset_index(name="n_selected_features")
        if not selection.empty
        else pd.DataFrame()
    )
    lines = [
        "## Results: Curve-Only Functional Current Severity",
        "",
        "This analysis tests the strong claim that angular reflectance curves alone can predict current plot severity.",
        "",
        "### Model Comparison",
        "",
        markdown_table(display[display_cols].round(4).sort_values("rmse"), max_rows=20),
        "",
        "### Paired Delta Versus Existing Nadir",
        "",
        "Positive RMSE reduction means the curve-only candidate improves over the compact nadir current-severity baseline on matched 2025 plot-week rows.",
        "",
        markdown_table(delta.round(4), max_rows=20),
        "",
        "### Feature Construction Audit",
        "",
        markdown_table(audit.round(4), max_rows=30),
        "",
        "### Selected Feature Counts",
        "",
        markdown_table(selected_summary, max_rows=30),
        "",
        "**Interpretation**: A defensible curve-only claim requires external 2025 improvement and no large optimism gap between grouped 2024 OOF and 2025. The log-curve variants include the vertical reflectance level because magnitude is part of the function; the shape variants remove that level by using near-nadir log-ratios.",
        "",
        "### Reproducibility",
        "",
        "- Training year: 2024.",
        "- Test year: 2025.",
        "- Target: same-week plot-level `ds_plot`.",
        "- Inputs: sampled VZA reflectance functions and reliable VZA-by-RAA reflectance functions.",
        "- Models: grouped 2024 stability-selected hurdle Ridge with top-20 and top-50 feature caps.",
        "- Excluded predictors: compact engineered summaries, treatment, cultivar, block, inoculation/design metadata, disease history, and residual correction from another model.",
        f"- Log: `{log_path}`",
        "",
        "### Outputs",
        "",
    ]
    lines.extend([f"- {label}: `{path}`" for label, path in paths.items()])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    total_started = time.perf_counter()
    configure_reused_pipeline_paths()
    for directory in [RESULTS_DIR, REPORTS_DIR, PREDICTIONS_DIR, FIGURES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging()

    long_2024, long_2025, disease_2024, disease_2025 = read_inputs()
    raa_2024 = read_raa(RAA_2024)
    raa_2025 = read_raa(RAA_2025)
    train_sources, test_sources = make_curve_sources(long_2024, long_2025, raa_2024, raa_2025)

    variants = {
        "curve_only_vza_log": (["vza"], ("log",)),
        "curve_only_vza_shape": (["vza"], ("shape",)),
        "curve_only_vza_log_shape": (["vza"], ("log", "shape")),
        "curve_only_vza_raa_log": (["vza", "vza_raa_reliable"], ("log",)),
        "curve_only_vza_raa_log_shape": (["vza", "vza_raa_reliable"], ("log", "shape")),
    }
    result_rows: list[dict[str, object]] = []
    prediction_map: dict[tuple[str, str], pd.DataFrame] = {}
    audits: list[pd.DataFrame] = []
    selections: list[pd.DataFrame] = []
    for feature_set, (sources, transforms) in variants.items():
        train_features, test_features, audit = build_sampled_curve_features(
            train_sources, test_sources, sources, transforms
        )
        rows, preds, selection = evaluate_feature_set(
            train_features,
            test_features,
            disease_2024,
            disease_2025,
            feature_set,
        )
        result_rows.extend(rows)
        prediction_map.update(preds)
        audits.append(audit.assign(feature_set=feature_set))
        selections.append(selection)

    context_predictions = load_context_predictions()
    context_rows = [
        score_prediction_frame(pred, model, feature_set, "existing_context")
        for (model, feature_set), pred in context_predictions.items()
    ]
    comparison = pd.concat(
        [pd.DataFrame(context_rows), pd.DataFrame(result_rows)],
        ignore_index=True,
        sort=False,
    ).sort_values("rmse")
    all_predictions = {**context_predictions, **prediction_map}
    delta = paired_delta_vs_nadir(all_predictions)
    audit_df = pd.concat(audits, ignore_index=True)
    selection_df = pd.concat(selections, ignore_index=True)

    paths = {
        "model_comparison": RESULTS_DIR / "curve_only_functional_model_comparison.csv",
        "paired_delta_vs_nadir": RESULTS_DIR / "curve_only_functional_delta_vs_nadir.csv",
        "feature_audit": RESULTS_DIR / "curve_only_functional_feature_audit.csv",
        "selected_features": RESULTS_DIR / "curve_only_functional_selected_features.csv",
        "predictions": PREDICTIONS_DIR,
    }
    comparison.to_csv(paths["model_comparison"], index=False)
    delta.to_csv(paths["paired_delta_vs_nadir"], index=False)
    audit_df.to_csv(paths["feature_audit"], index=False)
    selection_df.to_csv(paths["selected_features"], index=False)
    report_path = write_report(comparison, delta, audit_df, selection_df, paths, log_path)
    logging.info("Report: %s", report_path)
    log_phase("total", total_started)


if __name__ == "__main__":
    main()
