#!/usr/bin/env python3
"""Current severity from direct ElasticNet and sparse additive GAM curve models.

This tests two explainable alternatives to the current curve-only hurdle Ridge:

1. direct ElasticNetCV on sampled VZA log-reflectance curve values;
2. sparse additive spline-GAM on interpretable VZA curve-summary features.

Both are trained on 2024, audited with grouped 2024 OOF by plot, and externally
validated on 2025.  No treatment, cultivar, block, inoculation metadata, disease
history, or residual correction is used.
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import SplineTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[3]
MPLCONFIG_DIR = ROOT / "outputs/.matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.severity import analyze_current_plot_severity_2024_to_2025 as current_severity
from scripts.analysis.severity import debug_multiangular_rmse_bottleneck as residual_pipeline
from scripts.analysis.severity.analyze_current_severity_curve_only_functional_2024_to_2025 import (
    angle_columns,
    build_sampled_curve_features,
    make_curve_sources,
    merge_feature_frames,
)
from scripts.analysis.severity.analyze_current_severity_magnitude_shape_functional_2024_to_2025 import (
    COVARIATES,
    META_COLS,
    MIN_GROUP_ANGLES,
    paired_delta_vs_nadir,
    pivot_curves,
    read_inputs,
    score_prediction_frame,
    shifted_log_values,
)
from scripts.analysis.severity.analyze_current_severity_sparse_functional_discriminant_shape_2024_to_2025 import (
    RAA_2024,
    RAA_2025,
    read_raa,
)
from scripts.analysis.severity.analyze_multiangular_distribution_feature_family import (
    clean_token,
    markdown_table,
)

OUTPUT_ROOT = ROOT / "outputs/current_severity_curve_only_elasticnet_gam_2024_to_2025"
RESULTS_DIR = OUTPUT_ROOT / "results"
REPORTS_DIR = OUTPUT_ROOT / "reports"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = OUTPUT_ROOT / "figures"
LOGS_DIR = ROOT / "outputs/logs"

CURRENT_RESULTS_DIR = ROOT / "outputs/current_severity_2024_to_2025/results"
CURVE_ONLY_RESULTS_DIR = ROOT / "outputs/current_severity_curve_only_functional_2024_to_2025/results"

TARGET = current_severity.TARGET
SEED = current_severity.SEED
EPS = 1e-8


@dataclass
class GamTransformer:
    feature_names: list[str]
    imputer: SimpleImputer
    splines: list[SplineTransformer | None]
    expanded_names: list[str]
    scaler: StandardScaler


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"analyze_current_severity_curve_only_elasticnet_gam_2024_to_2025_{timestamp}.log"
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
    residual_pipeline.FROZEN_MANIFEST_PATH = OUTPUT_ROOT / "curve_only_elasticnet_gam_manifest.json"


def regression_scores(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": math.sqrt(mean_squared_error(y_true, pred)),
        "mae": mean_absolute_error(y_true, pred),
        "r2": r2_score(y_true, pred) if len(np.unique(y_true)) > 1 else math.nan,
        "spearman": residual_pipeline.safe_spearman(y_true, pred),
    }


def cv_splits(groups: np.ndarray, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    n_splits = min(5, len(np.unique(groups)))
    return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y, groups))


def fit_elasticnet_predict(
    train_x: pd.DataFrame | np.ndarray,
    test_x: pd.DataFrame | np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, ElasticNetCV, SimpleImputer, StandardScaler]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(train_x))
    x_test = scaler.transform(imputer.transform(test_x))
    model = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9, 1.0],
        alphas=np.logspace(-2, 2, 35),
        cv=cv_splits(groups, y_train),
        max_iter=50000,
        random_state=SEED,
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return pred, model, imputer, scaler


def grouped_oof_elasticnet(train_x: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    pred = np.zeros(len(y), dtype=float)
    for fit_idx, eval_idx in cv_splits(groups, y):
        fold_pred, _, _, _ = fit_elasticnet_predict(
            train_x.iloc[fit_idx],
            train_x.iloc[eval_idx],
            y[fit_idx],
            groups[fit_idx],
        )
        pred[eval_idx] = fold_pred
    return pred


def prediction_frame_from_values(
    test_aligned: pd.DataFrame,
    pred: np.ndarray,
    model: str,
    feature_set: str,
) -> pd.DataFrame:
    clipped = residual_pipeline.clip_predictions(pred, pred * 0 + test_aligned[TARGET].to_numpy(float))
    # Reclip below with training range in caller when available; this fallback keeps schema only.
    out = test_aligned[["plot_id", "predictor_week", "target_week"]].copy()
    out["model"] = model
    out["feature_set"] = feature_set
    out["covariates"] = COVARIATES
    out["y_true"] = test_aligned[TARGET].to_numpy(float)
    out["y_pred"] = clipped
    return out


def fit_direct_elasticnet(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    disease_2024: pd.DataFrame,
    disease_2025: pd.DataFrame,
    feature_set: str,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    started = time.perf_counter()
    train = current_severity.build_current_model_table(train_features, disease_2024)
    test = current_severity.build_current_model_table(test_features, disease_2025)
    cols, train_aligned, test_aligned = residual_pipeline.prepare_aligned(train, test)
    y_train = train_aligned[TARGET].to_numpy(float)
    groups = train_aligned["plot_id"].to_numpy()
    pred, model, _, _ = fit_elasticnet_predict(
        train_aligned[cols], test_aligned[cols], y_train, groups
    )
    in_sample, _, _, _ = fit_elasticnet_predict(
        train_aligned[cols], train_aligned[cols], y_train, groups
    )
    oof = grouped_oof_elasticnet(train_aligned[cols], y_train, groups)
    pred = residual_pipeline.clip_predictions(pred, y_train)
    in_sample = residual_pipeline.clip_predictions(in_sample, y_train)
    oof = residual_pipeline.clip_predictions(oof, y_train)
    model_name = "direct_elasticnet_curve_regression"
    predictions = residual_pipeline.prediction_frame(test_aligned, pred, model_name, feature_set)
    residual_pipeline.save_predictions(predictions, model_name, feature_set, COVARIATES)
    result = residual_pipeline.score_predictions(
        predictions, len(train_aligned), len(cols), model_name, feature_set
    )
    in_scores = regression_scores(y_train, in_sample)
    oof_scores = regression_scores(y_train, oof)
    result["source"] = "curve_only_direct_elasticnet"
    result["fit_time_s"] = time.perf_counter() - started
    result["elasticnet_alpha"] = float(model.alpha_)
    result["elasticnet_l1_ratio"] = float(model.l1_ratio_)
    result["n_nonzero_coefficients"] = int(np.count_nonzero(np.abs(model.coef_) > EPS))
    result["train_in_sample_rmse"] = in_scores["rmse"]
    result["train_grouped_oof_rmse"] = oof_scores["rmse"]
    result["train_oof_minus_in_sample_rmse"] = oof_scores["rmse"] - in_scores["rmse"]
    result["external_minus_oof_rmse"] = result["rmse"] - oof_scores["rmse"]
    audit = pd.DataFrame(
        [
            {
                "model": model_name,
                "feature_set": feature_set,
                "alpha": model.alpha_,
                "l1_ratio": model.l1_ratio_,
                "n_features": len(cols),
                "n_nonzero_coefficients": result["n_nonzero_coefficients"],
            }
        ]
    )
    return result, predictions, audit


def curve_summary_features_for_group(
    train_group: pd.DataFrame,
    test_group: pd.DataFrame,
    source: str,
    group: str,
    cols: list[str],
    angles: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    imputer = SimpleImputer(strategy="median")
    train_values = imputer.fit_transform(train_group[cols])
    test_values = imputer.transform(test_group[cols])
    train_log, test_log, shift = shifted_log_values(train_values, test_values)
    prefix = f"curvesummary__{clean_token(source)}__{clean_token(group)}"

    def summaries(values: np.ndarray) -> dict[str, np.ndarray]:
        centered = angles - angles.mean()
        denom = float(np.sum(centered**2))
        gradient = np.gradient(values, angles, axis=1)
        curvature = np.gradient(gradient, angles, axis=1)
        low = values[:, angles <= 25]
        high = values[:, angles >= 45]
        first = values[:, 0]
        last = values[:, -1]
        return {
            "mean": values.mean(axis=1),
            "nadir": first,
            "offnadir_last": last,
            "auc": np.trapezoid(values, x=angles, axis=1),
            "range": values.max(axis=1) - values.min(axis=1),
            "std": values.std(axis=1),
            "slope": np.sum((values - values.mean(axis=1, keepdims=True)) * centered, axis=1)
            / denom,
            "high_minus_low": high.mean(axis=1) - low.mean(axis=1),
            "last_minus_nadir": last - first,
            "mean_abs_derivative": np.abs(gradient).mean(axis=1),
            "mean_abs_curvature": np.abs(curvature).mean(axis=1),
        }

    train_out = train_group[META_COLS].reset_index(drop=True).copy()
    test_out = test_group[META_COLS].reset_index(drop=True).copy()
    train_summaries = summaries(train_log)
    test_summaries = summaries(test_log)
    for name, values in train_summaries.items():
        train_out[f"{prefix}__{name}"] = values
    for name, values in test_summaries.items():
        test_out[f"{prefix}__{name}"] = values
    audit = {
        "source": source,
        "curve_group": group,
        "n_train_curves": len(train_group),
        "n_test_curves": len(test_group),
        "n_angles": len(cols),
        "n_summary_features": len(train_summaries),
        "shift_for_log": shift,
    }
    return train_out, test_out, audit


def build_curve_summary_features(
    train_sources: dict[str, pd.DataFrame],
    test_sources: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    started = time.perf_counter()
    train_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    source = "vza"
    train_pivot = train_sources[source]
    test_pivot = test_sources[source]
    for group in sorted(set(train_pivot["curve_group"]).intersection(test_pivot["curve_group"])):
        train_group = train_pivot[train_pivot["curve_group"].eq(group)].copy()
        test_group = test_pivot[test_pivot["curve_group"].eq(group)].copy()
        cols, angles = angle_columns(train_group, test_group)
        if len(cols) < MIN_GROUP_ANGLES:
            continue
        train_out, test_out, audit = curve_summary_features_for_group(
            train_group, test_group, source, group, cols, angles
        )
        train_frames.append(train_out)
        test_frames.append(test_out)
        audits.append(audit)
    train = merge_feature_frames(train_frames)
    test = merge_feature_frames(test_frames)
    audit = pd.DataFrame(audits)
    logging.info(
        "curve summary features: train=%d rows/%d features test=%d rows/%d features",
        len(train),
        train.shape[1] - len(META_COLS),
        len(test),
        test.shape[1] - len(META_COLS),
    )
    log_phase("build curve summary features", started)
    return train, test, audit


def fit_gam_transformer(train_x: pd.DataFrame, test_x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, GamTransformer]:
    feature_names = list(train_x.columns)
    imputer = SimpleImputer(strategy="median")
    train_imp = imputer.fit_transform(train_x)
    test_imp = imputer.transform(test_x)
    train_blocks: list[np.ndarray] = []
    test_blocks: list[np.ndarray] = []
    splines: list[SplineTransformer | None] = []
    expanded_names: list[str] = []
    for idx, feature in enumerate(feature_names):
        train_col = train_imp[:, [idx]]
        test_col = test_imp[:, [idx]]
        if float(np.nanstd(train_col)) <= EPS:
            splines.append(None)
            continue
        spline = SplineTransformer(
            n_knots=4,
            degree=3,
            include_bias=False,
            extrapolation="continue",
        )
        train_basis = spline.fit_transform(train_col)
        test_basis = spline.transform(test_col)
        train_blocks.append(train_basis)
        test_blocks.append(test_basis)
        splines.append(spline)
        expanded_names.extend([f"{feature}__spline_{j:02d}" for j in range(train_basis.shape[1])])
    train_basis_all = np.hstack(train_blocks)
    test_basis_all = np.hstack(test_blocks)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_basis_all)
    test_scaled = scaler.transform(test_basis_all)
    transformer = GamTransformer(feature_names, imputer, splines, expanded_names, scaler)
    return train_scaled, test_scaled, transformer


def transform_gam(transformer: GamTransformer, x: pd.DataFrame) -> np.ndarray:
    values = transformer.imputer.transform(x[transformer.feature_names])
    blocks: list[np.ndarray] = []
    for idx, spline in enumerate(transformer.splines):
        if spline is None:
            continue
        blocks.append(spline.transform(values[:, [idx]]))
    return transformer.scaler.transform(np.hstack(blocks))


def select_top_correlated_features(
    train_x: pd.DataFrame,
    y_train: np.ndarray,
    top_k: int = 35,
) -> list[str]:
    imputer = SimpleImputer(strategy="median")
    values = imputer.fit_transform(train_x)
    scores: list[tuple[float, str]] = []
    for idx, col in enumerate(train_x.columns):
        x = values[:, idx]
        if float(np.std(x)) <= EPS or float(np.std(y_train)) <= EPS:
            score = 0.0
        else:
            score = abs(float(np.corrcoef(x, y_train)[0, 1]))
            if not np.isfinite(score):
                score = 0.0
        scores.append((score, col))
    return [col for _, col in sorted(scores, reverse=True)[: min(top_k, len(scores))]]


def fit_gam_predict(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    y_train: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, RidgeCV, GamTransformer, list[str]]:
    selected_cols = select_top_correlated_features(train_x, y_train, top_k=35)
    train_basis, test_basis, transformer = fit_gam_transformer(
        train_x[selected_cols], test_x[selected_cols]
    )
    model = RidgeCV(
        alphas=np.logspace(-2, 4, 40),
        cv=cv_splits(groups, y_train),
    )
    model.fit(train_basis, y_train)
    return model.predict(test_basis), model, transformer, selected_cols


def grouped_oof_gam(train_x: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    pred = np.zeros(len(y), dtype=float)
    for fit_idx, eval_idx in cv_splits(groups, y):
        fold_pred, _, _, _ = fit_gam_predict(
            train_x.iloc[fit_idx],
            train_x.iloc[eval_idx],
            y[fit_idx],
            groups[fit_idx],
        )
        pred[eval_idx] = fold_pred
    return pred


def fit_sparse_additive_gam(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    disease_2024: pd.DataFrame,
    disease_2025: pd.DataFrame,
    feature_set: str,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    started = time.perf_counter()
    train = current_severity.build_current_model_table(train_features, disease_2024)
    test = current_severity.build_current_model_table(test_features, disease_2025)
    cols, train_aligned, test_aligned = residual_pipeline.prepare_aligned(train, test)
    y_train = train_aligned[TARGET].to_numpy(float)
    groups = train_aligned["plot_id"].to_numpy()
    pred, model, transformer, selected_cols = fit_gam_predict(
        train_aligned[cols], test_aligned[cols], y_train, groups
    )
    in_sample_basis = transform_gam(transformer, train_aligned[selected_cols])
    in_sample = model.predict(in_sample_basis)
    oof = grouped_oof_gam(train_aligned[cols], y_train, groups)
    pred = residual_pipeline.clip_predictions(pred, y_train)
    in_sample = residual_pipeline.clip_predictions(in_sample, y_train)
    oof = residual_pipeline.clip_predictions(oof, y_train)
    model_name = "sparse_additive_spline_gam"
    predictions = residual_pipeline.prediction_frame(test_aligned, pred, model_name, feature_set)
    residual_pipeline.save_predictions(predictions, model_name, feature_set, COVARIATES)
    result = residual_pipeline.score_predictions(
        predictions, len(train_aligned), len(cols), model_name, feature_set
    )
    in_scores = regression_scores(y_train, in_sample)
    oof_scores = regression_scores(y_train, oof)
    coef = np.asarray(model.coef_, dtype=float).ravel()
    nonzero = np.flatnonzero(np.abs(coef) > EPS)
    selected_terms = [transformer.expanded_names[idx] for idx in nonzero]
    result["source"] = "curve_summary_sparse_gam"
    result["fit_time_s"] = time.perf_counter() - started
    result["ridge_alpha"] = float(model.alpha_)
    result["selected_summary_features"] = len(selected_cols)
    result["n_spline_basis_terms"] = len(transformer.expanded_names)
    result["n_nonzero_coefficients"] = len(selected_terms)
    result["train_in_sample_rmse"] = in_scores["rmse"]
    result["train_grouped_oof_rmse"] = oof_scores["rmse"]
    result["train_oof_minus_in_sample_rmse"] = oof_scores["rmse"] - in_scores["rmse"]
    result["external_minus_oof_rmse"] = result["rmse"] - oof_scores["rmse"]
    audit = pd.DataFrame(
        {
            "model": model_name,
            "feature_set": feature_set,
            "basis_term": selected_terms,
            "coefficient": coef[nonzero],
        }
    ).sort_values("coefficient", key=lambda s: np.abs(s), ascending=False)
    return result, predictions, audit


def load_context_predictions() -> dict[tuple[str, str], pd.DataFrame]:
    paths = {
        (
            "current_hurdle_top20_raw_positive",
            "compact_anomaly_nadir",
        ): CURRENT_RESULTS_DIR
        / "predictions/severity_predictions_current_hurdle_top20_raw_positive_compact_anomaly_nadir_spectral_plus_week.csv",
        (
            "current_hurdle_stability_top50_raw_positive",
            "compact_anomaly_multiangular",
        ): CURRENT_RESULTS_DIR
        / "predictions/severity_predictions_current_hurdle_stability_top50_raw_positive_compact_anomaly_multiangular_spectral_plus_week.csv",
        (
            "current_hurdle_stability_top50_raw_positive",
            "curve_only_vza_log",
        ): CURVE_ONLY_RESULTS_DIR
        / "predictions/severity_predictions_current_hurdle_stability_top50_raw_positive_curve_only_vza_log_spectral_plus_week.csv",
    }
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for key, path in paths.items():
        if path.exists():
            out[key] = pd.read_csv(path)
        else:
            logging.warning("Missing context prediction: %s", path)
    return out


def write_report(
    comparison: pd.DataFrame,
    delta: pd.DataFrame,
    model_audit: pd.DataFrame,
    summary_audit: pd.DataFrame,
    paths: dict[str, Path],
    log_path: Path,
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "curve_only_elasticnet_gam_current_severity_summary.md"
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
        "elasticnet_alpha",
        "elasticnet_l1_ratio",
        "n_nonzero_coefficients",
    ]
    display = comparison.copy()
    for col in display_cols:
        if col not in display.columns:
            display[col] = math.nan
    lines = [
        "## Results: Direct ElasticNet and Sparse Additive GAM Current Severity",
        "",
        "This analysis tests direct ElasticNet on sampled VZA log curves and a sparse additive spline-GAM on curve summaries.",
        "",
        "### Model Comparison",
        "",
        markdown_table(display[display_cols].round(4).sort_values("rmse"), max_rows=20),
        "",
        "### Paired Delta Versus Existing Nadir",
        "",
        markdown_table(delta.round(4), max_rows=20),
        "",
        "### Model Audit",
        "",
        markdown_table(model_audit.round(4), max_rows=30),
        "",
        "### Curve Summary Feature Audit",
        "",
        markdown_table(summary_audit.round(4), max_rows=30),
        "",
        "**Interpretation**: Direct ElasticNet is the clean linear explainable model on curve samples. The sparse additive GAM allows nonlinear univariate effects of engineered curve summaries, but the grouped OOF and external 2025 gap determine whether that flexibility is credible.",
        "",
        "### Reproducibility",
        "",
        "- Training year: 2024.",
        "- Test year: 2025.",
        "- Target: same-week plot-level `ds_plot`.",
        "- Inputs: VZA log-reflectance curve samples for ElasticNet; VZA log-curve summaries for sparse additive GAM.",
        "- Validation: grouped 2024 OOF by plot and external 2025 test.",
        "- Excluded predictors: treatment, cultivar, block, inoculation/design metadata, disease history, RAA, and residual correction.",
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

    sampled_train, sampled_test, _ = build_sampled_curve_features(
        train_sources, test_sources, ["vza"], ("log",)
    )
    summary_train, summary_test, summary_audit = build_curve_summary_features(
        train_sources, test_sources
    )
    en_result, en_pred, en_audit = fit_direct_elasticnet(
        sampled_train,
        sampled_test,
        disease_2024,
        disease_2025,
        "direct_elasticnet_vza_log_curve_samples",
    )
    gam_result, gam_pred, gam_audit = fit_sparse_additive_gam(
        summary_train,
        summary_test,
        disease_2024,
        disease_2025,
        "sparse_gam_vza_log_curve_summaries",
    )

    context_predictions = load_context_predictions()
    context_rows = [
        score_prediction_frame(pred, model, feature_set, "existing_context")
        for (model, feature_set), pred in context_predictions.items()
    ]
    new_predictions = {
        (en_result["model"], en_result["feature_set"]): en_pred,
        (gam_result["model"], gam_result["feature_set"]): gam_pred,
    }
    comparison = pd.concat(
        [pd.DataFrame(context_rows), pd.DataFrame([en_result, gam_result])],
        ignore_index=True,
        sort=False,
    ).sort_values("rmse")
    delta = paired_delta_vs_nadir({**context_predictions, **new_predictions})
    model_audit = pd.concat([en_audit, gam_audit], ignore_index=True, sort=False)

    paths = {
        "model_comparison": RESULTS_DIR / "curve_only_elasticnet_gam_model_comparison.csv",
        "paired_delta_vs_nadir": RESULTS_DIR / "curve_only_elasticnet_gam_delta_vs_nadir.csv",
        "model_audit": RESULTS_DIR / "curve_only_elasticnet_gam_model_audit.csv",
        "summary_feature_audit": RESULTS_DIR / "curve_only_elasticnet_gam_summary_feature_audit.csv",
        "predictions": PREDICTIONS_DIR,
    }
    comparison.to_csv(paths["model_comparison"], index=False)
    delta.to_csv(paths["paired_delta_vs_nadir"], index=False)
    model_audit.to_csv(paths["model_audit"], index=False)
    summary_audit.to_csv(paths["summary_feature_audit"], index=False)
    report_path = write_report(comparison, delta, model_audit, summary_audit, paths, log_path)
    logging.info("Report: %s", report_path)
    log_phase("total", total_started)


if __name__ == "__main__":
    main()
