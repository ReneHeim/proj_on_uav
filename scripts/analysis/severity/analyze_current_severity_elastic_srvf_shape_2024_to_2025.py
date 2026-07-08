#!/usr/bin/env python3
"""Current severity from elastic SRVF angular-shape features.

This script is the first Fisher-Rao/SRVF-inspired test of the angular-curve
hypothesis.  For each plot-week and reflectance curve over VZA, it builds:

* magnitude summaries from the original reflectance curve;
* elastic shape features from the square-root velocity function (SRVF) of the
  log-ratio angular curve;
* derivative-energy features and magnitude-by-shape interactions.

No cultivar, treatment, block, inoculation/design metadata, RAA, or disease
history predictors are used.
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
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
MPLCONFIG_DIR = ROOT / "outputs/.matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.severity import analyze_current_plot_severity_2024_to_2025 as current_severity
from scripts.analysis.severity import debug_multiangular_rmse_bottleneck as residual_pipeline
from scripts.analysis.severity.analyze_current_severity_magnitude_shape_functional_2024_to_2025 import (
    COVARIATES,
    META_COLS,
    MIN_GROUP_ANGLES,
    angle_columns,
    curve_magnitude_features,
    load_context_predictions,
    merge_feature_frames,
    paired_delta_vs_nadir,
    pivot_curves,
    read_inputs,
    score_prediction_frame,
    shifted_log_values,
    standardize_pair,
)
from scripts.analysis.severity.analyze_multiangular_distribution_feature_family import (
    clean_token,
    markdown_table,
)

OUTPUT_ROOT = ROOT / "outputs/current_severity_elastic_srvf_shape_2024_to_2025"
RESULTS_DIR = OUTPUT_ROOT / "results"
REPORTS_DIR = OUTPUT_ROOT / "reports"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = OUTPUT_ROOT / "figures"
LOGS_DIR = ROOT / "outputs/logs"

TARGET = current_severity.TARGET
SEED = current_severity.SEED
ALPHAS = current_severity.ALPHAS
N_SRVF_COMPONENTS = 2
EPS = 1e-8


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"analyze_current_severity_elastic_srvf_shape_2024_to_2025_{timestamp}.log"
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
    residual_pipeline.FROZEN_MANIFEST_PATH = OUTPUT_ROOT / "elastic_srvf_shape_manifest.json"


def trapz_mean(values: np.ndarray, angles: np.ndarray) -> np.ndarray:
    span = float(np.max(angles) - np.min(angles))
    if span <= 0:
        return values.mean(axis=1)
    return np.trapezoid(values, x=angles, axis=1) / span


def srvf_transform(shape: np.ndarray, angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    derivative = np.gradient(shape, angles, axis=1)
    srvf = np.sign(derivative) * np.sqrt(np.abs(derivative) + EPS)
    norm = np.sqrt(np.trapezoid(srvf**2, x=angles, axis=1))
    norm = np.where(norm <= EPS, 1.0, norm)
    return srvf, srvf / norm[:, None]


def srvf_shape_features(
    train_unit: np.ndarray,
    test_unit: np.ndarray,
    angles: np.ndarray,
    prefix: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    train_out: dict[str, np.ndarray] = {}
    test_out: dict[str, np.ndarray] = {}
    for idx, angle in enumerate(angles):
        name = f"{prefix}__elastic_shape_q_{angle:04.1f}"
        train_out[name] = train_unit[:, idx]
        test_out[name] = test_unit[:, idx]

    low = angles <= 25
    high = angles >= 45
    train_out[f"{prefix}__elastic_shape_mean"] = train_unit.mean(axis=1)
    test_out[f"{prefix}__elastic_shape_mean"] = test_unit.mean(axis=1)
    train_out[f"{prefix}__elastic_shape_std"] = train_unit.std(axis=1)
    test_out[f"{prefix}__elastic_shape_std"] = test_unit.std(axis=1)
    train_out[f"{prefix}__elastic_shape_positive_fraction"] = (train_unit > 0).mean(axis=1)
    test_out[f"{prefix}__elastic_shape_positive_fraction"] = (test_unit > 0).mean(axis=1)
    train_out[f"{prefix}__elastic_shape_high_minus_low"] = (
        train_unit[:, high].mean(axis=1) - train_unit[:, low].mean(axis=1)
    )
    test_out[f"{prefix}__elastic_shape_high_minus_low"] = (
        test_unit[:, high].mean(axis=1) - test_unit[:, low].mean(axis=1)
    )
    train_out[f"{prefix}__elastic_shape_roughness"] = np.abs(np.diff(train_unit, axis=1)).mean(axis=1)
    test_out[f"{prefix}__elastic_shape_roughness"] = np.abs(np.diff(test_unit, axis=1)).mean(axis=1)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_unit)
    test_scaled = scaler.transform(test_unit)
    n_components = min(N_SRVF_COMPONENTS, train_scaled.shape[1], train_scaled.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=SEED)
    train_scores = pca.fit_transform(train_scaled)
    test_scores = pca.transform(test_scaled)
    for idx in range(n_components):
        name = f"{prefix}__elastic_shape_fpca_{idx + 1}"
        train_out[name] = train_scores[:, idx]
        test_out[name] = test_scores[:, idx]

    audit = {
        "n_elastic_shape_components": n_components,
        "elastic_shape_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
    }
    return train_out, test_out, audit


def srvf_energy_features(
    train_srvf: np.ndarray,
    test_srvf: np.ndarray,
    angles: np.ndarray,
    prefix: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    low = angles <= 25
    high = angles >= 45
    train_energy = train_srvf**2
    test_energy = test_srvf**2
    train_total = np.trapezoid(train_energy, x=angles, axis=1)
    test_total = np.trapezoid(test_energy, x=angles, axis=1)
    train_pos = trapz_mean(np.where(train_srvf > 0, train_energy, 0.0), angles)
    test_pos = trapz_mean(np.where(test_srvf > 0, test_energy, 0.0), angles)
    train_neg = trapz_mean(np.where(train_srvf < 0, train_energy, 0.0), angles)
    test_neg = trapz_mean(np.where(test_srvf < 0, test_energy, 0.0), angles)
    train_out = {
        f"{prefix}__elastic_energy_total": train_total,
        f"{prefix}__elastic_energy_log_total": np.log1p(train_total),
        f"{prefix}__elastic_energy_positive": train_pos,
        f"{prefix}__elastic_energy_negative": train_neg,
        f"{prefix}__elastic_energy_high_minus_low": train_energy[:, high].mean(axis=1)
        - train_energy[:, low].mean(axis=1),
    }
    test_out = {
        f"{prefix}__elastic_energy_total": test_total,
        f"{prefix}__elastic_energy_log_total": np.log1p(test_total),
        f"{prefix}__elastic_energy_positive": test_pos,
        f"{prefix}__elastic_energy_negative": test_neg,
        f"{prefix}__elastic_energy_high_minus_low": test_energy[:, high].mean(axis=1)
        - test_energy[:, low].mean(axis=1),
    }
    return train_out, test_out


def interaction_features(
    train_features: dict[str, np.ndarray],
    test_features: dict[str, np.ndarray],
    prefix: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    mag_names = [
        f"{prefix}__mag_log_mean",
        f"{prefix}__mag_log_nadir",
        f"{prefix}__mag_auc",
    ]
    shape_names = [
        f"{prefix}__elastic_shape_fpca_1",
        f"{prefix}__elastic_shape_fpca_2",
        f"{prefix}__elastic_shape_high_minus_low",
        f"{prefix}__elastic_energy_log_total",
    ]
    train_out: dict[str, np.ndarray] = {}
    test_out: dict[str, np.ndarray] = {}
    for mag_name in mag_names:
        if mag_name not in train_features:
            continue
        train_mag, test_mag = standardize_pair(train_features[mag_name], test_features[mag_name])
        mag_short = mag_name.split("__")[-1]
        for shape_name in shape_names:
            if shape_name not in train_features:
                continue
            train_shape, test_shape = standardize_pair(
                train_features[shape_name], test_features[shape_name]
            )
            shape_short = shape_name.split("__")[-1]
            name = f"{prefix}__interaction_{mag_short}_x_{shape_short}"
            train_out[name] = train_mag * train_shape
            test_out[name] = test_mag * test_shape
    return train_out, test_out


def build_group_features(
    train_group: pd.DataFrame,
    test_group: pd.DataFrame,
    group: str,
    angle_cols: list[str],
    angles: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    train_meta = train_group[META_COLS].reset_index(drop=True)
    test_meta = test_group[META_COLS].reset_index(drop=True)
    imputer = SimpleImputer(strategy="median")
    train_values = imputer.fit_transform(train_group[angle_cols])
    test_values = imputer.transform(test_group[angle_cols])
    train_log, test_log, shift = shifted_log_values(train_values, test_values)
    train_shape = train_log - train_log[:, [0]]
    test_shape = test_log - test_log[:, [0]]
    train_srvf, train_unit = srvf_transform(train_shape, angles)
    test_srvf, test_unit = srvf_transform(test_shape, angles)

    prefix = f"srvf__{clean_token(group)}"
    train_mag = curve_magnitude_features(train_values, train_log, angles, prefix)
    test_mag = curve_magnitude_features(test_values, test_log, angles, prefix)
    train_shape_features, test_shape_features, pca_audit = srvf_shape_features(
        train_unit, test_unit, angles, prefix
    )
    train_energy, test_energy = srvf_energy_features(train_srvf, test_srvf, angles, prefix)
    train_interactions, test_interactions = interaction_features(
        {**train_mag, **train_shape_features, **train_energy},
        {**test_mag, **test_shape_features, **test_energy},
        prefix,
    )

    train_all = {**train_mag, **train_shape_features, **train_energy, **train_interactions}
    test_all = {**test_mag, **test_shape_features, **test_energy, **test_interactions}
    train_out = pd.concat([train_meta, pd.DataFrame(train_all)], axis=1)
    test_out = pd.concat([test_meta, pd.DataFrame(test_all)], axis=1)
    audit = {
        "curve_group": group,
        "n_train_curves": len(train_group),
        "n_test_curves": len(test_group),
        "n_angles": len(angle_cols),
        "shift_for_log": shift,
        "n_magnitude_features": len(train_mag),
        "n_elastic_shape_features": len(train_shape_features),
        "n_elastic_energy_features": len(train_energy),
        "n_interaction_features": len(train_interactions),
        "n_total_features": len(train_all),
        **pca_audit,
    }
    return train_out, test_out, audit


def build_elastic_features(
    long_2024: pd.DataFrame,
    long_2025: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    started = time.perf_counter()
    train_pivot = pivot_curves(long_2024)
    test_pivot = pivot_curves(long_2025)
    train_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    common_groups = sorted(set(train_pivot["curve_group"]).intersection(test_pivot["curve_group"]))
    for group in common_groups:
        train_group = train_pivot[train_pivot["curve_group"].eq(group)].copy()
        test_group = test_pivot[test_pivot["curve_group"].eq(group)].copy()
        cols, angles = angle_columns(train_group, test_group)
        if len(cols) < MIN_GROUP_ANGLES:
            continue
        train_features, test_features, audit = build_group_features(
            train_group, test_group, group, cols, angles
        )
        train_frames.append(train_features)
        test_frames.append(test_features)
        audits.append(audit)
    if not train_frames:
        raise RuntimeError("No elastic SRVF features were created.")
    train = merge_feature_frames(train_frames)
    test = merge_feature_frames(test_frames)
    audit = pd.DataFrame(audits).sort_values("curve_group")
    logging.info(
        "elastic SRVF features: train=%d rows test=%d rows features=%d groups=%d",
        len(train),
        len(test),
        len([col for col in train.columns if col not in META_COLS]),
        len(audit),
    )
    log_phase("build elastic SRVF features", started)
    return train, test, audit


def feature_variants(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    all_cols = [col for col in train_features.columns if col not in META_COLS]
    mag_cols = [col for col in all_cols if "__mag_" in col]
    shape_cols = [col for col in all_cols if "__elastic_shape_" in col]
    energy_cols = [col for col in all_cols if "__elastic_energy_" in col]
    interaction_cols = [col for col in all_cols if "__interaction_" in col]
    variants = {
        "elastic_srvf_shape_only": shape_cols,
        "elastic_srvf_shape_energy": shape_cols + energy_cols,
        "elastic_srvf_magnitude_shape": mag_cols + shape_cols + energy_cols,
        "elastic_srvf_magnitude_shape_interaction": (
            mag_cols + shape_cols + energy_cols + interaction_cols
        ),
    }
    out = {}
    for name, cols in variants.items():
        out[name] = (
            train_features[META_COLS + cols].copy(),
            test_features[META_COLS + cols].copy(),
        )
        logging.info("%s: %d feature columns", name, len(cols))
    return out


def fit_fixed_hurdle_ridge(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_set: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    cols, train_aligned, test_aligned = residual_pipeline.prepare_aligned(train, test)
    y_train = train_aligned[TARGET].to_numpy(float)
    y_present = (y_train > 0).astype(int)
    fit_started = time.perf_counter()
    classifier = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegressionCV(
                    Cs=np.logspace(-2, 1, 8),
                    cv=3,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=SEED,
                ),
            ),
        ]
    )
    if len(np.unique(y_present)) < 2:
        disease_prob = np.full(len(test_aligned), float(y_present.mean()))
    else:
        classifier.fit(train_aligned[cols], y_present)
        disease_prob = classifier.predict_proba(test_aligned[cols])[:, 1]

    positive_mask = y_train > 0
    regressor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=ALPHAS)),
        ]
    )
    if positive_mask.sum() < 5:
        severity_pred = np.full(len(test_aligned), float(np.nanmean(y_train)))
    else:
        regressor.fit(train_aligned.loc[positive_mask, cols], y_train[positive_mask])
        severity_pred = regressor.predict(test_aligned[cols])

    pred = disease_prob * severity_pred
    zero_weeks = (
        train_aligned.groupby("target_week")[TARGET]
        .max()
        .loc[lambda values: values <= 0]
        .index.to_numpy()
    )
    if zero_weeks.size:
        pred[np.isin(test_aligned["target_week"].to_numpy(), zero_weeks)] = 0.0
    pred = residual_pipeline.clip_predictions(pred, y_train)
    model = "fixed_elastic_srvf_hurdle_ridge"
    predictions = residual_pipeline.prediction_frame(test_aligned, pred, model, feature_set)
    predictions["disease_probability"] = disease_prob
    residual_pipeline.save_predictions(predictions, model, feature_set, COVARIATES)
    result = residual_pipeline.score_predictions(
        predictions, len(train_aligned), len(cols), model, feature_set
    )
    result["source"] = "elastic_srvf_shape"
    result["fit_time_s"] = time.perf_counter() - fit_started
    result["zero_target_weeks_from_2024"] = ",".join(map(str, zero_weeks.tolist()))
    return result, predictions


def evaluate_fixed_variants(
    variants: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    disease_2024: pd.DataFrame,
    disease_2025: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    results: list[dict[str, object]] = []
    predictions: dict[tuple[str, str], pd.DataFrame] = {}
    started = time.perf_counter()
    for feature_set, (train_features, test_features) in variants.items():
        train = current_severity.build_current_model_table(train_features, disease_2024)
        test = current_severity.build_current_model_table(test_features, disease_2025)
        logging.info("%s model table: train=%d test=%d", feature_set, len(train), len(test))
        result, pred = fit_fixed_hurdle_ridge(train, test, feature_set)
        results.append(result)
        predictions[(result["model"], feature_set)] = pred
    log_phase("fit fixed elastic SRVF models", started)
    return pd.DataFrame(results).sort_values("rmse"), predictions


def evaluate_sparse_variants(
    variants: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    disease_2024: pd.DataFrame,
    disease_2025: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame], pd.DataFrame]:
    selected = {
        "sparse_elastic_srvf_shape_energy_top50": variants["elastic_srvf_shape_energy"],
        "sparse_elastic_srvf_magnitude_shape_top50": variants["elastic_srvf_magnitude_shape"],
    }
    results: list[dict[str, object]] = []
    predictions: dict[tuple[str, str], pd.DataFrame] = {}
    selections: list[pd.DataFrame] = []
    started = time.perf_counter()
    for feature_set, (train_features, test_features) in selected.items():
        train = current_severity.build_current_model_table(train_features, disease_2024)
        test = current_severity.build_current_model_table(test_features, disease_2025)
        result, pred, selection = current_severity.current_hurdle_stability_topk_model(
            train,
            test,
            feature_set,
            top_k=50,
            log_positive=False,
        )
        result["source"] = "elastic_srvf_shape_sparse"
        results.append(result)
        predictions[(result["model"], feature_set)] = pred
        selections.append(selection)
    log_phase("fit sparse elastic SRVF models", started)
    selection_df = pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()
    return pd.DataFrame(results).sort_values("rmse"), predictions, selection_df


def write_report(
    comparison: pd.DataFrame,
    delta: pd.DataFrame,
    week_summary: pd.DataFrame,
    audit: pd.DataFrame,
    paths: dict[str, Path],
    log_path: Path,
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "elastic_srvf_shape_current_severity_summary.md"
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
    ]
    lines = [
        "## Results: Elastic SRVF Shape Current Severity Model",
        "",
        "This analysis tests a Fisher-Rao/SRVF-inspired angular-shape encoding for current plot severity.",
        "",
        "### Model Comparison",
        "",
        markdown_table(comparison[display_cols].round(4).sort_values("rmse"), max_rows=20),
        "",
        "### Paired Delta Versus Existing Nadir",
        "",
        "Positive RMSE reduction means the candidate improves over the existing compact nadir current-severity baseline on matched 2025 plot-week rows.",
        "",
        markdown_table(delta.round(4), max_rows=20),
        "",
        "### Target-Week Diagnostics",
        "",
        markdown_table(week_summary.round(4), max_rows=60),
        "",
        "### Feature Construction Audit",
        "",
        markdown_table(audit.round(4), max_rows=30),
        "",
        "**Interpretation**: The shape-only row tests whether normalized angular SRVF shape carries severity signal without absolute reflectance magnitude. The magnitude-shape rows test whether elastic angular shape complements magnitude. The sparse rows use 2024-only grouped feature selection.",
        "",
        "### Reproducibility",
        "",
        "- Training year: 2024.",
        "- Test year: 2025.",
        "- Target: same-week plot-level `ds_plot`.",
        "- Inputs: VZA curves from cached multiangular reflectance distribution features.",
        "- Shape transform: shifted log reflectance, near-nadir log-ratio normalization, SRVF derivative transform, and unit SRVF normalization.",
        "- Excluded predictors: treatment, cultivar, block, inoculation/design metadata, RAA, and disease history.",
        "- Models: fixed hurdle Ridge for main ablations; grouped 2024-only top-50 stability-selected hurdle Ridge for sparse refinements.",
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
    train_features, test_features, audit = build_elastic_features(long_2024, long_2025)
    variants = feature_variants(train_features, test_features)
    fixed_results, fixed_predictions = evaluate_fixed_variants(
        variants, disease_2024, disease_2025
    )
    sparse_results, sparse_predictions, sparse_selection = evaluate_sparse_variants(
        variants, disease_2024, disease_2025
    )
    elastic_results = pd.concat([fixed_results, sparse_results], ignore_index=True, sort=False)

    context_predictions = load_context_predictions()
    context_rows = [
        score_prediction_frame(pred, model, feature_set, "existing_context")
        for (model, feature_set), pred in context_predictions.items()
    ]
    comparison = pd.concat(
        [pd.DataFrame(context_rows), elastic_results],
        ignore_index=True,
        sort=False,
    ).sort_values("rmse")
    all_predictions = {**context_predictions, **fixed_predictions, **sparse_predictions}
    delta = paired_delta_vs_nadir(all_predictions)
    week_summary = current_severity.prediction_week_summary(all_predictions)

    paths = {
        "model_comparison": RESULTS_DIR / "elastic_srvf_shape_model_comparison.csv",
        "paired_delta_vs_nadir": RESULTS_DIR / "elastic_srvf_shape_delta_vs_nadir.csv",
        "target_week_summary": RESULTS_DIR / "elastic_srvf_shape_target_week_summary.csv",
        "feature_audit": RESULTS_DIR / "elastic_srvf_shape_feature_audit.csv",
        "sparse_selected_features": RESULTS_DIR / "elastic_srvf_shape_sparse_selected_features.csv",
        "predictions": PREDICTIONS_DIR,
    }
    comparison.to_csv(paths["model_comparison"], index=False)
    delta.to_csv(paths["paired_delta_vs_nadir"], index=False)
    week_summary.to_csv(paths["target_week_summary"], index=False)
    audit.to_csv(paths["feature_audit"], index=False)
    sparse_selection.to_csv(paths["sparse_selected_features"], index=False)
    report_path = write_report(comparison, delta, week_summary, audit, paths, log_path)
    logging.info("Report: %s", report_path)
    log_phase("total", total_started)


if __name__ == "__main__":
    main()
