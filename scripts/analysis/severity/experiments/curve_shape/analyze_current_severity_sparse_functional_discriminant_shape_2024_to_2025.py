#!/usr/bin/env python3
"""Current severity from sparse functional-discriminant angular shape.

This analysis tests whether disease is encoded in sparse, supervised angular
shape directions rather than in all VZA bins.  It learns two low-dimensional
scores per curve group from 2024 only:

* a sparse disease-presence direction;
* a sparse continuous-severity direction.

The directions are fitted on log-ratio angular curves, so the main signal is
curve shape relative to near-nadir.  A nested grouped 2024 OOF audit refits the
shape directions inside each fold to diagnose overfitting.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

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
from scripts.analysis.severity.experiments.curve_shape.analyze_current_severity_curve_embeddings_2024_to_2025 import (
    ANGLE_GRID,
    CURVE_BANDS,
    clean_band_name,
)
from scripts.analysis.severity.experiments.curve_shape.analyze_current_severity_magnitude_shape_functional_2024_to_2025 import (
    COVARIATES,
    META_COLS,
    MIN_GROUP_ANGLES,
    angle_columns,
    load_context_predictions,
    paired_delta_vs_nadir,
    pivot_curves,
    read_inputs,
    score_prediction_frame,
    shifted_log_values,
)
from scripts.analysis.severity.experiments.geometry.analyze_current_severity_raa_geometry_fusion_2024_to_2025 import (
    MIN_ANGULAR_BIN_IMAGES,
    MIN_ANGULAR_BIN_PIXELS,
    RAA_2024,
    RAA_2025,
)
from scripts.analysis.severity.analyze_multiangular_distribution_feature_family import (
    clean_token,
    markdown_table,
)

OUTPUT_ROOT = ROOT / "outputs/runs/analysis/severity/current/sparse_functional_discriminant_shape_2024_to_2025"
RESULTS_DIR = OUTPUT_ROOT / "results"
REPORTS_DIR = OUTPUT_ROOT / "reports"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = OUTPUT_ROOT / "figures"
LOGS_DIR = ROOT / "outputs/archive/legacy_unscoped/logs"

TARGET = current_severity.TARGET
SEED = current_severity.SEED
MIN_TRAIN_CURVES_PER_GROUP = 120
EPS = 1e-8


@dataclass
class SparseDirection:
    imputer: SimpleImputer
    scaler: StandardScaler
    coef: np.ndarray
    intercept: float
    score_mean: float
    score_std: float
    alpha: float
    l1_ratio: float


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = (
        LOGS_DIR
        / f"analyze_current_severity_sparse_functional_discriminant_shape_2024_to_2025_{timestamp}.log"
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
    residual_pipeline.FROZEN_MANIFEST_PATH = (
        OUTPUT_ROOT / "sparse_functional_discriminant_shape_manifest.json"
    )


def read_raa(path: Path) -> pd.DataFrame:
    started = time.perf_counter()
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pl.read_parquet(path).to_pandas()
    logging.info("Read RAA %s rows x %s cols from %s", frame.shape[0], frame.shape[1], path)
    log_phase(f"read {path.name}", started)
    return frame


def filter_reliable_raa(raa: pd.DataFrame) -> pd.DataFrame:
    return raa[
        (raa["n_pixels"] >= MIN_ANGULAR_BIN_PIXELS) & (raa["n_images"] >= MIN_ANGULAR_BIN_IMAGES)
    ].copy()


def pivot_raa_vza_curves(raa: pd.DataFrame, reliable_only: bool) -> pd.DataFrame:
    data = filter_reliable_raa(raa) if reliable_only else raa.copy()
    data["band_token"] = data["band_name"].map(clean_band_name)
    data = data[data["band_token"].isin(CURVE_BANDS)].copy()
    data["raa_token"] = data["raa_class"].map(clean_token)
    data["curve_group"] = data["band_token"] + "__raa_" + data["raa_token"]
    pivot = data.pivot_table(
        index=META_COLS + ["curve_group"],
        columns="vza_midpoint",
        values="reflectance",
        aggfunc="mean",
    ).reset_index()
    pivot.columns.name = None
    angle_cols = [col for col in pivot.columns if isinstance(col, (int, float))]
    return pivot.rename(columns={angle: f"angle_{float(angle):04.1f}" for angle in angle_cols})


def disease_lookup(disease: pd.DataFrame) -> pd.DataFrame:
    return disease[["plot_id", "week", "ds_plot"]].rename(columns={"ds_plot": TARGET}).copy()


def add_target(frame: pd.DataFrame, disease: pd.DataFrame) -> pd.DataFrame:
    return frame.merge(disease_lookup(disease), on=["plot_id", "week"], how="inner")


def shape_values(
    train_group: pd.DataFrame,
    apply_group: pd.DataFrame,
    angle_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    train_values = imputer.fit_transform(train_group[angle_cols])
    apply_values = imputer.transform(apply_group[angle_cols])
    train_log, apply_log, _ = shifted_log_values(train_values, apply_values)
    return train_log - train_log[:, [0]], apply_log - apply_log[:, [0]]


def grouped_cv_splits(
    y: np.ndarray, groups: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]] | int:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        return 3
    n_splits = min(5, len(unique_groups))
    return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y, groups))


def fit_sparse_direction(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> SparseDirection:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(imputer.fit_transform(x))
    if len(np.unique(y)) < 2 or float(np.nanstd(y)) <= EPS:
        coef = np.zeros(x_scaled.shape[1], dtype=float)
        score = np.zeros(x_scaled.shape[0], dtype=float)
        return SparseDirection(imputer, scaler, coef, 0.0, 0.0, 1.0, math.nan, math.nan)
    model = ElasticNetCV(
        l1_ratio=[0.7, 0.9, 1.0],
        alphas=np.logspace(-3, 1, 30),
        cv=grouped_cv_splits(y, groups),
        max_iter=20000,
        random_state=SEED,
        n_jobs=None,
    )
    model.fit(x_scaled, y)
    coef = np.asarray(model.coef_, dtype=float)
    score = x_scaled @ coef
    score_mean = float(np.mean(score))
    score_std = float(np.std(score))
    if score_std <= EPS:
        score_std = 1.0
    return SparseDirection(
        imputer=imputer,
        scaler=scaler,
        coef=coef,
        intercept=float(model.intercept_),
        score_mean=score_mean,
        score_std=score_std,
        alpha=float(model.alpha_),
        l1_ratio=float(model.l1_ratio_),
    )


def transform_direction(direction: SparseDirection, x: np.ndarray) -> np.ndarray:
    x_scaled = direction.scaler.transform(direction.imputer.transform(x))
    score = x_scaled @ direction.coef
    return (score - direction.score_mean) / direction.score_std


def build_group_sparse_scores(
    train_group: pd.DataFrame,
    apply_group: pd.DataFrame,
    disease_train: pd.DataFrame,
    source: str,
    group: str,
    angle_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_targeted = add_target(train_group, disease_train)
    if len(train_targeted) < MIN_TRAIN_CURVES_PER_GROUP:
        return pd.DataFrame(), pd.DataFrame()
    train_shape, apply_shape = shape_values(train_targeted, apply_group, angle_cols)
    y = train_targeted[TARGET].to_numpy(float)
    groups = train_targeted["plot_id"].to_numpy()
    presence_direction = fit_sparse_direction(train_shape, (y > 0).astype(float), groups)
    severity_direction = fit_sparse_direction(train_shape, y, groups)
    prefix = f"sfd__{source}__{clean_token(group)}"
    apply_out = apply_group[META_COLS].reset_index(drop=True).copy()
    apply_out[f"{prefix}__presence_score"] = transform_direction(presence_direction, apply_shape)
    apply_out[f"{prefix}__severity_score"] = transform_direction(severity_direction, apply_shape)
    train_out = train_targeted[META_COLS].reset_index(drop=True).copy()
    train_out[f"{prefix}__presence_score"] = transform_direction(presence_direction, train_shape)
    train_out[f"{prefix}__severity_score"] = transform_direction(severity_direction, train_shape)
    audit = pd.DataFrame(
        [
            {
                "source": source,
                "curve_group": group,
                "role": "presence",
                "n_train_curves": len(train_targeted),
                "n_apply_curves": len(apply_group),
                "n_angles": len(angle_cols),
                "n_nonzero_coefficients": int(
                    np.count_nonzero(np.abs(presence_direction.coef) > EPS)
                ),
                "alpha": presence_direction.alpha,
                "l1_ratio": presence_direction.l1_ratio,
            },
            {
                "source": source,
                "curve_group": group,
                "role": "severity",
                "n_train_curves": len(train_targeted),
                "n_apply_curves": len(apply_group),
                "n_angles": len(angle_cols),
                "n_nonzero_coefficients": int(
                    np.count_nonzero(np.abs(severity_direction.coef) > EPS)
                ),
                "alpha": severity_direction.alpha,
                "l1_ratio": severity_direction.l1_ratio,
            },
        ]
    )
    return train_out, apply_out, audit


def merge_feature_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=META_COLS)
    merged = frames[0]
    for frame in frames[1:]:
        feature_cols = [col for col in frame.columns if col not in META_COLS]
        merged = merged.merge(frame[META_COLS + feature_cols], on=META_COLS, how="outer")
    return merged


def build_sparse_discriminant_features_for_sources(
    train_sources: dict[str, pd.DataFrame],
    apply_sources: dict[str, pd.DataFrame],
    disease_train: pd.DataFrame,
    selected_sources: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_frames: list[pd.DataFrame] = []
    apply_frames: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    for source in selected_sources:
        train_pivot = train_sources[source]
        apply_pivot = apply_sources[source]
        common_groups = sorted(
            set(train_pivot["curve_group"]).intersection(apply_pivot["curve_group"])
        )
        for group in common_groups:
            train_group = train_pivot[train_pivot["curve_group"].eq(group)].copy()
            apply_group = apply_pivot[apply_pivot["curve_group"].eq(group)].copy()
            cols, _ = angle_columns(train_group, apply_group)
            if len(cols) < MIN_GROUP_ANGLES:
                continue
            train_out, apply_out, audit = build_group_sparse_scores(
                train_group, apply_group, disease_train, source, group, cols
            )
            if train_out.empty:
                continue
            train_frames.append(train_out)
            apply_frames.append(apply_out)
            audits.append(audit)
    train = merge_feature_frames(train_frames)
    apply = merge_feature_frames(apply_frames)
    audit = pd.concat(audits, ignore_index=True) if audits else pd.DataFrame()
    return train, apply, audit


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
    for name in train:
        logging.info(
            "%s source: train=%d rows/%d groups, test=%d rows/%d groups",
            name,
            len(train[name]),
            train[name]["curve_group"].nunique(),
            len(test[name]),
            test[name]["curve_group"].nunique(),
        )
    return train, test


def fit_hurdle_all_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_set: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    cols, train_aligned, test_aligned = residual_pipeline.prepare_aligned(train, test)
    fit_started = time.perf_counter()
    pred = current_severity.fit_hurdle_with_columns(
        train_aligned, test_aligned, cols, cols, log_positive=False
    )
    model = "sparse_functional_discriminant_hurdle_ridge"
    predictions = residual_pipeline.prediction_frame(test_aligned, pred, model, feature_set)
    residual_pipeline.save_predictions(predictions, model, feature_set, COVARIATES)
    result = residual_pipeline.score_predictions(
        predictions, len(train_aligned), len(cols), model, feature_set
    )
    in_sample_pred = current_severity.fit_hurdle_with_columns(
        train_aligned, train_aligned, cols, cols, log_positive=False
    )
    in_sample = current_severity.regression_scores(
        train_aligned[TARGET].to_numpy(float), in_sample_pred
    )
    result["source"] = "sparse_functional_discriminant_shape"
    result["fit_time_s"] = time.perf_counter() - fit_started
    result["train_in_sample_rmse"] = in_sample["rmse"]
    return result, predictions


def nested_oof_predictions(
    train_sources: dict[str, pd.DataFrame],
    disease_2024: pd.DataFrame,
    selected_sources: list[str],
    feature_set: str,
) -> pd.DataFrame:
    base = train_sources[selected_sources[0]][META_COLS].drop_duplicates().copy()
    base = current_severity.build_current_model_table(base, disease_2024)
    groups = base["plot_id"].to_numpy()
    preds: list[pd.DataFrame] = []
    n_splits = min(5, len(np.unique(groups)))
    for fold, (fit_idx, eval_idx) in enumerate(
        GroupKFold(n_splits=n_splits).split(base, base[TARGET], groups), start=1
    ):
        fit_keys = base.iloc[fit_idx][["plot_id", "week"]]
        eval_keys = base.iloc[eval_idx][["plot_id", "week"]]
        fold_train_sources: dict[str, pd.DataFrame] = {}
        fold_eval_sources: dict[str, pd.DataFrame] = {}
        for source in selected_sources:
            source_frame = train_sources[source]
            fold_train_sources[source] = source_frame.merge(
                fit_keys, on=["plot_id", "week"], how="inner"
            )
            fold_eval_sources[source] = source_frame.merge(
                eval_keys, on=["plot_id", "week"], how="inner"
            )
        fold_train_features, fold_eval_features, _ = build_sparse_discriminant_features_for_sources(
            fold_train_sources, fold_eval_sources, disease_2024, selected_sources
        )
        train_table = current_severity.build_current_model_table(fold_train_features, disease_2024)
        eval_table = current_severity.build_current_model_table(fold_eval_features, disease_2024)
        cols, train_aligned, eval_aligned = residual_pipeline.prepare_aligned(
            train_table, eval_table
        )
        pred = current_severity.fit_hurdle_with_columns(
            train_aligned, eval_aligned, cols, cols, log_positive=False
        )
        frame = residual_pipeline.prediction_frame(
            eval_aligned,
            pred,
            "nested_oof_sparse_functional_discriminant_hurdle_ridge",
            feature_set,
        )
        frame["fold"] = fold
        preds.append(frame)
    return pd.concat(preds, ignore_index=True).sort_values(["target_week", "plot_id"])


def evaluate_variant(
    train_sources: dict[str, pd.DataFrame],
    test_sources: dict[str, pd.DataFrame],
    disease_2024: pd.DataFrame,
    disease_2025: pd.DataFrame,
    feature_set: str,
    selected_sources: list[str],
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    started = time.perf_counter()
    train_features, test_features, audit = build_sparse_discriminant_features_for_sources(
        train_sources, test_sources, disease_2024, selected_sources
    )
    logging.info(
        "%s sparse discriminant features: train=%d rows/%d features test=%d rows/%d features",
        feature_set,
        len(train_features),
        train_features.shape[1] - len(META_COLS),
        len(test_features),
        test_features.shape[1] - len(META_COLS),
    )
    train_table = current_severity.build_current_model_table(train_features, disease_2024)
    test_table = current_severity.build_current_model_table(test_features, disease_2025)
    result, external_pred = fit_hurdle_all_features(train_table, test_table, feature_set)
    oof_pred = nested_oof_predictions(train_sources, disease_2024, selected_sources, feature_set)
    residual_pipeline.save_predictions(
        oof_pred,
        "nested_oof_sparse_functional_discriminant_hurdle_ridge",
        feature_set,
        COVARIATES,
    )
    oof_scores = current_severity.regression_scores(
        oof_pred["y_true"].to_numpy(float), oof_pred["y_pred"].to_numpy(float)
    )
    result["train_nested_grouped_oof_rmse"] = oof_scores["rmse"]
    result["train_oof_minus_in_sample_rmse"] = (
        result["train_nested_grouped_oof_rmse"] - result["train_in_sample_rmse"]
    )
    result["external_minus_nested_oof_rmse"] = (
        result["rmse"] - result["train_nested_grouped_oof_rmse"]
    )
    result["nested_oof_mae"] = oof_scores["mae"]
    result["nested_oof_r2"] = oof_scores["r2"]
    result["nested_oof_spearman"] = oof_scores["spearman"]
    result["build_and_fit_time_s"] = time.perf_counter() - started
    return result, external_pred, oof_pred, audit


def write_report(
    comparison: pd.DataFrame,
    delta: pd.DataFrame,
    audit: pd.DataFrame,
    paths: dict[str, Path],
    log_path: Path,
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "sparse_functional_discriminant_shape_current_severity_summary.md"
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
        "train_nested_grouped_oof_rmse",
        "external_minus_nested_oof_rmse",
    ]
    display = comparison.copy()
    for col in display_cols:
        if col not in display.columns:
            display[col] = math.nan
    lines = [
        "## Results: Sparse Functional-Discriminant Shape Current Severity",
        "",
        "This analysis learns sparse disease-supervised angular shape directions from 2024 reflectance curves and validates them on 2025.",
        "",
        "### Model Comparison",
        "",
        markdown_table(display[display_cols].round(4).sort_values("rmse"), max_rows=20),
        "",
        "### Paired Delta Versus Existing Nadir",
        "",
        "Positive RMSE reduction means the candidate improves over the existing compact nadir current-severity baseline on matched 2025 plot-week rows.",
        "",
        markdown_table(delta.round(4), max_rows=20),
        "",
        "### Sparse Direction Audit",
        "",
        markdown_table(audit.round(4), max_rows=40),
        "",
        "**Interpretation**: The nested grouped OOF RMSE is the key overfitting check because each fold refits the sparse functional directions without seeing the held-out 2024 plots. A large gap between in-sample and nested OOF, or between nested OOF and external 2025, indicates that the learned shape directions are unstable across plots or years.",
        "",
        "### Reproducibility",
        "",
        "- Training year: 2024.",
        "- Test year: 2025.",
        "- Target: same-week plot-level `ds_plot`.",
        "- Inputs: VZA log-ratio curves and reliable VZA-by-RAA log-ratio curves.",
        "- Feature extractor: per-curve-group ElasticNetCV sparse functional directions for disease presence and continuous severity.",
        "- Overfitting audit: nested GroupKFold by `plot_id`; sparse directions are refit inside each fold.",
        "- Excluded predictors: treatment, cultivar, block, inoculation/design metadata, and disease history.",
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
        "sparse_flda_shape_vza": ["vza"],
        "sparse_flda_shape_vza_raa": ["vza", "vza_raa_reliable"],
    }
    results: list[dict[str, object]] = []
    predictions: dict[tuple[str, str], pd.DataFrame] = {}
    audits: list[pd.DataFrame] = []
    for feature_set, selected_sources in variants.items():
        result, external_pred, oof_pred, audit = evaluate_variant(
            train_sources,
            test_sources,
            disease_2024,
            disease_2025,
            feature_set,
            selected_sources,
        )
        results.append(result)
        predictions[(result["model"], feature_set)] = external_pred
        audits.append(audit.assign(feature_set=feature_set))

    context_predictions = load_context_predictions()
    context_rows = [
        score_prediction_frame(pred, model, feature_set, "existing_context")
        for (model, feature_set), pred in context_predictions.items()
    ]
    comparison = pd.concat(
        [pd.DataFrame(context_rows), pd.DataFrame(results)],
        ignore_index=True,
        sort=False,
    ).sort_values("rmse")
    all_predictions = {**context_predictions, **predictions}
    delta = paired_delta_vs_nadir(all_predictions)
    audit_df = pd.concat(audits, ignore_index=True) if audits else pd.DataFrame()

    paths = {
        "model_comparison": RESULTS_DIR
        / "sparse_functional_discriminant_shape_model_comparison.csv",
        "paired_delta_vs_nadir": RESULTS_DIR
        / "sparse_functional_discriminant_shape_delta_vs_nadir.csv",
        "sparse_direction_audit": RESULTS_DIR
        / "sparse_functional_discriminant_shape_direction_audit.csv",
        "predictions": PREDICTIONS_DIR,
    }
    comparison.to_csv(paths["model_comparison"], index=False)
    delta.to_csv(paths["paired_delta_vs_nadir"], index=False)
    audit_df.to_csv(paths["sparse_direction_audit"], index=False)
    report_path = write_report(comparison, delta, audit_df, paths, log_path)
    logging.info("Report: %s", report_path)
    log_phase("total", total_started)


if __name__ == "__main__":
    main()
