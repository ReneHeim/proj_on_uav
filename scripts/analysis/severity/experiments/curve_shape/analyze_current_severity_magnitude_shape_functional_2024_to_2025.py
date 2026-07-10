#!/usr/bin/env python3
"""Current severity from joint magnitude-shape functional curve embeddings.

This script tests a fixed version of the leaf-angle hypothesis:

    disease severity is not only reflectance magnitude and not only angular
    shape; it may be encoded in their coupling.

For each plot-week curve over view zenith angle, the script builds three
imaging-only components:

* magnitude: absolute curve level summaries;
* shape: normalized angular log-ratio curve summaries and FPCA scores;
* interaction: standardized magnitude terms multiplied by shape scores.

The same hurdle Ridge model is then run on four predeclared ablations:
magnitude only, shape only, magnitude+shape, and magnitude+shape+interaction.
No cultivar, treatment, block, inoculation, RAA, or disease-history predictors
are used.
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
    INPUT_RESULTS_DIR,
    OSAVI_METRICS,
    REFLECTANCE_METRICS,
    clean_band_name,
)
from scripts.analysis.severity.analyze_multiangular_distribution_feature_family import (
    clean_token,
    markdown_table,
)

OUTPUT_ROOT = ROOT / "outputs/runs/analysis/severity/current/magnitude_shape_functional_2024_to_2025"
RESULTS_DIR = OUTPUT_ROOT / "results"
REPORTS_DIR = OUTPUT_ROOT / "reports"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = OUTPUT_ROOT / "figures"
LOGS_DIR = ROOT / "outputs/archive/legacy_unscoped/logs"

DISEASE_2024_CLEAN = ROOT / "outputs/shared/disease/clean_disease_scores_2024.csv"
DISEASE_2025_CLEAN = ROOT / "outputs/shared/disease/clean_disease_scores_2025.csv"
CURRENT_RESULTS_DIR = ROOT / "outputs/runs/analysis/severity/current/2024_to_2025/results"
CURVE_RESULTS_DIR = ROOT / "outputs/runs/analysis/severity/current/curve_embeddings_2024_to_2025/results"

COVARIATES = "spectral_plus_week"
META_COLS = ["year", "week", "plot_id", "cult", "trt"]
MIN_GROUP_ANGLES = 5
N_SHAPE_COMPONENTS = 2
BLOCK_PCA_VARIANCE = 0.90
MAX_MAGNITUDE_BLOCK_COMPONENTS = 8
MAX_SHAPE_BLOCK_COMPONENTS = 8
INTERACTION_BLOCK_COMPONENTS = 3
EPS = 1e-4

TARGET = current_severity.TARGET
TARGET_LOG = current_severity.TARGET_LOG
WARNING_TARGET = current_severity.WARNING_TARGET
WARNING_THRESHOLD = current_severity.WARNING_THRESHOLD
SEED = current_severity.SEED
ALPHAS = current_severity.ALPHAS


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = (
        LOGS_DIR
        / f"analyze_current_severity_magnitude_shape_functional_2024_to_2025_{timestamp}.log"
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
        OUTPUT_ROOT / "magnitude_shape_functional_manifest.json"
    )


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    started = time.perf_counter()
    long_2024 = pd.read_csv(INPUT_RESULTS_DIR / "distribution_features_long_2024.csv")
    long_2025 = pd.read_csv(INPUT_RESULTS_DIR / "distribution_features_long_2025.csv")
    disease_2024 = pd.read_csv(DISEASE_2024_CLEAN)
    disease_2025 = pd.read_csv(DISEASE_2025_CLEAN)
    log_phase("read cached long features and disease scores", started)
    return long_2024, long_2025, disease_2024, disease_2025


def filter_curve_rows(long: pd.DataFrame) -> pd.DataFrame:
    data = long.copy()
    data["band_token"] = data["band_name"].map(clean_band_name)
    data["metric_token"] = data["metric"].map(clean_token)
    keep = data["band_token"].isin(CURVE_BANDS) & (
        data["metric_token"].isin(REFLECTANCE_METRICS) | data["metric_token"].isin(OSAVI_METRICS)
    )
    keep &= (data["band_token"] != "osavi") | data["metric_token"].isin(OSAVI_METRICS)
    keep &= (data["band_token"] == "osavi") | data["metric_token"].isin(REFLECTANCE_METRICS)
    data = data.loc[keep].copy()
    data["curve_group"] = data["band_token"] + "__" + data["metric_token"]
    return data


def pivot_curves(long: pd.DataFrame) -> pd.DataFrame:
    data = filter_curve_rows(long)
    pivot = data.pivot_table(
        index=META_COLS + ["curve_group"],
        columns="vza_midpoint",
        values="value",
        aggfunc="mean",
    ).reset_index()
    pivot.columns.name = None
    angle_cols = [col for col in pivot.columns if isinstance(col, (int, float))]
    return pivot.rename(columns={angle: f"angle_{float(angle):04.1f}" for angle in angle_cols})


def angle_columns(train: pd.DataFrame, test: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    cols = []
    for angle in ANGLE_GRID:
        col = f"angle_{angle:04.1f}"
        if col in train.columns and col in test.columns:
            cols.append(col)
    angles = np.asarray([float(col.replace("angle_", "")) for col in cols], dtype=float)
    return cols, angles


def shifted_log_values(
    train_values: np.ndarray,
    test_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    train_min = float(np.nanmin(train_values))
    shift = max(EPS, EPS - train_min)
    train_log = np.log(np.clip(train_values + shift, EPS, None))
    test_log = np.log(np.clip(test_values + shift, EPS, None))
    return train_log, test_log, shift


def curve_magnitude_features(
    values: np.ndarray, log_values: np.ndarray, angles: np.ndarray, prefix: str
) -> dict[str, np.ndarray]:
    low = values[:, angles <= 25]
    high = values[:, angles >= 45]
    off_nadir = values[:, 1:] if values.shape[1] > 1 else values
    return {
        f"{prefix}__mag_mean": values.mean(axis=1),
        f"{prefix}__mag_log_mean": log_values.mean(axis=1),
        f"{prefix}__mag_nadir": values[:, 0],
        f"{prefix}__mag_log_nadir": log_values[:, 0],
        f"{prefix}__mag_offnadir_mean": off_nadir.mean(axis=1),
        f"{prefix}__mag_high_mean": high.mean(axis=1),
        f"{prefix}__mag_auc": np.trapezoid(values, x=angles, axis=1),
        f"{prefix}__mag_l2": np.sqrt(np.mean(values**2, axis=1)),
        f"{prefix}__mag_high_minus_low_abs": high.mean(axis=1) - low.mean(axis=1),
    }


def shape_summary_features(
    shape: np.ndarray, angles: np.ndarray, prefix: str
) -> dict[str, np.ndarray]:
    low = shape[:, angles <= 25]
    high = shape[:, angles >= 45]
    centered = angles - angles.mean()
    denom = float(np.sum(centered**2))
    slope = np.sum((shape - shape.mean(axis=1, keepdims=True)) * centered, axis=1) / denom
    quadratic = np.column_stack([np.ones_like(angles), centered, centered**2])
    coef = np.linalg.pinv(quadratic) @ shape.T
    return {
        f"{prefix}__shape_range": np.max(shape, axis=1) - np.min(shape, axis=1),
        f"{prefix}__shape_std": np.std(shape, axis=1),
        f"{prefix}__shape_high_minus_low": high.mean(axis=1) - low.mean(axis=1),
        f"{prefix}__shape_signed_auc": np.trapezoid(shape, x=angles, axis=1),
        f"{prefix}__shape_slope": slope,
        f"{prefix}__shape_curvature": coef[2, :],
        f"{prefix}__shape_roughness": np.mean(np.abs(np.diff(shape, axis=1)), axis=1),
    }


def fit_shape_scores(
    train_shape: np.ndarray,
    test_shape: np.ndarray,
    prefix: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_shape)
    test_scaled = scaler.transform(test_shape)
    n_components = min(N_SHAPE_COMPONENTS, train_scaled.shape[1], train_scaled.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=SEED)
    train_scores = pca.fit_transform(train_scaled)
    test_scores = pca.transform(test_scaled)
    train_features = {
        f"{prefix}__shape_fpca_{idx + 1}": train_scores[:, idx] for idx in range(n_components)
    }
    test_features = {
        f"{prefix}__shape_fpca_{idx + 1}": test_scores[:, idx] for idx in range(n_components)
    }
    audit = {
        "n_shape_fpca_components": n_components,
        "shape_fpca_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
    }
    return train_features, test_features, audit


def standardize_pair(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = float(np.mean(train))
    std = float(np.std(train))
    if std <= EPS:
        std = 1.0
    return (train - mean) / std, (test - mean) / std


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
        f"{prefix}__shape_fpca_1",
        f"{prefix}__shape_fpca_2",
        f"{prefix}__shape_high_minus_low",
    ]
    train_out: dict[str, np.ndarray] = {}
    test_out: dict[str, np.ndarray] = {}
    for mag_name in mag_names:
        if mag_name not in train_features:
            continue
        mag_short = mag_name.split("__")[-1]
        train_mag, test_mag = standardize_pair(train_features[mag_name], test_features[mag_name])
        for shape_name in shape_names:
            if shape_name not in train_features:
                continue
            shape_short = shape_name.split("__")[-1]
            train_shape, test_shape = standardize_pair(
                train_features[shape_name], test_features[shape_name]
            )
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
    prefix = f"msf__{clean_token(group)}"

    train_mag = curve_magnitude_features(train_values, train_log, angles, prefix)
    test_mag = curve_magnitude_features(test_values, test_log, angles, prefix)
    train_shape_features = shape_summary_features(train_shape, angles, prefix)
    test_shape_features = shape_summary_features(test_shape, angles, prefix)
    train_scores, test_scores, pca_audit = fit_shape_scores(train_shape, test_shape, prefix)
    train_shape_features.update(train_scores)
    test_shape_features.update(test_scores)
    train_interactions, test_interactions = interaction_features(
        {**train_mag, **train_shape_features},
        {**test_mag, **test_shape_features},
        prefix,
    )

    train_all = {**train_mag, **train_shape_features, **train_interactions}
    test_all = {**test_mag, **test_shape_features, **test_interactions}
    train_out = pd.concat([train_meta, pd.DataFrame(train_all)], axis=1)
    test_out = pd.concat([test_meta, pd.DataFrame(test_all)], axis=1)
    audit = {
        "curve_group": group,
        "n_train_curves": len(train_group),
        "n_test_curves": len(test_group),
        "n_angles": len(angle_cols),
        "shift_for_log": shift,
        "n_magnitude_features": len(train_mag),
        "n_shape_features": len(train_shape_features),
        "n_interaction_features": len(train_interactions),
        "n_total_features": len(train_all),
        **pca_audit,
    }
    return train_out, test_out, audit


def merge_feature_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    merged = frames[0]
    for frame in frames[1:]:
        feature_cols = [col for col in frame.columns if col not in META_COLS]
        merged = merged.merge(frame[META_COLS + feature_cols], on=META_COLS, how="outer")
    return merged


def build_magnitude_shape_features(
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
        raise RuntimeError("No magnitude-shape functional features were created.")
    train = merge_feature_frames(train_frames)
    test = merge_feature_frames(test_frames)
    audit = pd.DataFrame(audits).sort_values("curve_group")
    logging.info(
        "magnitude-shape functional features: train=%d rows test=%d rows features=%d groups=%d",
        len(train),
        len(test),
        len([col for col in train.columns if col not in META_COLS]),
        len(audit),
    )
    log_phase("build magnitude-shape functional features", started)
    return train, test, audit


def feature_variants(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    all_cols = [col for col in train_features.columns if col not in META_COLS]
    mag_cols = [col for col in all_cols if "__mag_" in col]
    shape_cols = [col for col in all_cols if "__shape_" in col]
    interaction_cols = [col for col in all_cols if "__interaction_" in col]
    variants = {
        "functional_magnitude_only": mag_cols,
        "functional_shape_only": shape_cols,
        "functional_magnitude_shape": mag_cols + shape_cols,
        "functional_magnitude_shape_interaction": mag_cols + shape_cols + interaction_cols,
    }
    out = {}
    for name, cols in variants.items():
        out[name] = (
            train_features[META_COLS + cols].copy(),
            test_features[META_COLS + cols].copy(),
        )
        logging.info("%s: %d feature columns", name, len(cols))
    return out


def block_pca_features(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    columns: list[str],
    prefix: str,
    max_components: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_values = scaler.fit_transform(imputer.fit_transform(train_features[columns]))
    test_values = scaler.transform(imputer.transform(test_features[columns]))
    full_components = min(max_components, train_values.shape[1], train_values.shape[0] - 1)
    pca_full = PCA(n_components=full_components, random_state=SEED)
    pca_full.fit(train_values)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative, BLOCK_PCA_VARIANCE) + 1)
    n_components = max(1, min(n_components, full_components))
    pca = PCA(n_components=n_components, random_state=SEED)
    train_scores = pca.fit_transform(train_values)
    test_scores = pca.transform(test_values)
    train_out = pd.DataFrame(
        {f"{prefix}_pc{idx + 1}": train_scores[:, idx] for idx in range(n_components)},
        index=train_features.index,
    )
    test_out = pd.DataFrame(
        {f"{prefix}_pc{idx + 1}": test_scores[:, idx] for idx in range(n_components)},
        index=test_features.index,
    )
    audit = {
        f"{prefix}_source_features": len(columns),
        f"{prefix}_components": n_components,
        f"{prefix}_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
    }
    return train_out, test_out, audit


def standardized_product(
    train_left: np.ndarray,
    test_left: np.ndarray,
    train_right: np.ndarray,
    test_right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left_mean = float(np.mean(train_left))
    left_std = float(np.std(train_left)) or 1.0
    right_mean = float(np.mean(train_right))
    right_std = float(np.std(train_right)) or 1.0
    train_product = ((train_left - left_mean) / left_std) * ((train_right - right_mean) / right_std)
    test_product = ((test_left - left_mean) / left_std) * ((test_right - right_mean) / right_std)
    return train_product, test_product


def compact_block_variants(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> tuple[dict[str, tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame]:
    all_cols = [col for col in train_features.columns if col not in META_COLS]
    mag_cols = [col for col in all_cols if "__mag_" in col]
    shape_cols = [col for col in all_cols if "__shape_" in col]
    train_mag, test_mag, mag_audit = block_pca_features(
        train_features,
        test_features,
        mag_cols,
        "block_mag",
        MAX_MAGNITUDE_BLOCK_COMPONENTS,
    )
    train_shape, test_shape, shape_audit = block_pca_features(
        train_features,
        test_features,
        shape_cols,
        "block_shape",
        MAX_SHAPE_BLOCK_COMPONENTS,
    )
    train_block = pd.concat(
        [
            train_features[META_COLS].reset_index(drop=True),
            train_mag.reset_index(drop=True),
            train_shape.reset_index(drop=True),
        ],
        axis=1,
    )
    test_block = pd.concat(
        [
            test_features[META_COLS].reset_index(drop=True),
            test_mag.reset_index(drop=True),
            test_shape.reset_index(drop=True),
        ],
        axis=1,
    )

    train_interactions: dict[str, np.ndarray] = {}
    test_interactions: dict[str, np.ndarray] = {}
    mag_pc_cols = list(train_mag.columns)[:INTERACTION_BLOCK_COMPONENTS]
    shape_pc_cols = list(train_shape.columns)[:INTERACTION_BLOCK_COMPONENTS]
    for mag_col in mag_pc_cols:
        for shape_col in shape_pc_cols:
            name = f"block_interaction__{mag_col}_x_{shape_col}"
            train_interactions[name], test_interactions[name] = standardized_product(
                train_mag[mag_col].to_numpy(float),
                test_mag[mag_col].to_numpy(float),
                train_shape[shape_col].to_numpy(float),
                test_shape[shape_col].to_numpy(float),
            )

    train_block_interaction = pd.concat(
        [train_block, pd.DataFrame(train_interactions, index=train_block.index)],
        axis=1,
    )
    test_block_interaction = pd.concat(
        [test_block, pd.DataFrame(test_interactions, index=test_block.index)],
        axis=1,
    )
    variants = {
        "regularized_block_magnitude_shape": (train_block.copy(), test_block.copy()),
        "regularized_block_magnitude_shape_interaction": (
            train_block_interaction,
            test_block_interaction,
        ),
    }
    audit = pd.DataFrame(
        [
            {
                "variant": "regularized_block_magnitude_shape",
                **mag_audit,
                **shape_audit,
                "interaction_components_per_block": 0,
                "interaction_features": 0,
                "total_features": train_block.shape[1] - len(META_COLS),
            },
            {
                "variant": "regularized_block_magnitude_shape_interaction",
                **mag_audit,
                **shape_audit,
                "interaction_components_per_block": INTERACTION_BLOCK_COMPONENTS,
                "interaction_features": len(train_interactions),
                "total_features": train_block_interaction.shape[1] - len(META_COLS),
            },
        ]
    )
    for name, (train_variant, _) in variants.items():
        logging.info("%s: %d feature columns", name, train_variant.shape[1] - len(META_COLS))
    return variants, audit


def add_stage_week_interactions(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add acquisition-stage interactions using known week only.

    The interaction weight is standardized from 2024 weeks and applied to both
    years, so no 2025 labels affect the transform.
    """
    train_out = train_features.copy()
    test_out = test_features.copy()
    train_week = train_out["week"].to_numpy(float)
    test_week = test_out["week"].to_numpy(float)
    week_mean = float(train_week.mean())
    week_std = float(train_week.std())
    if week_std <= EPS:
        week_std = 1.0
    train_weight = (train_week - week_mean) / week_std
    test_weight = (test_week - week_mean) / week_std
    feature_cols = [col for col in train_features.columns if col not in META_COLS]
    train_stage = {
        f"stage_week_z__{col}": train_features[col].to_numpy(float) * train_weight
        for col in feature_cols
    }
    test_stage = {
        f"stage_week_z__{col}": test_features[col].to_numpy(float) * test_weight
        for col in feature_cols
    }
    train_out = pd.concat([train_out, pd.DataFrame(train_stage, index=train_out.index)], axis=1)
    test_out = pd.concat([test_out, pd.DataFrame(test_stage, index=test_out.index)], axis=1)
    return train_out, test_out


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
    model = "fixed_functional_hurdle_ridge"
    predictions = residual_pipeline.prediction_frame(test_aligned, pred, model, feature_set)
    predictions["disease_probability"] = disease_prob
    residual_pipeline.save_predictions(predictions, model, feature_set, COVARIATES)
    result = residual_pipeline.score_predictions(
        predictions, len(train_aligned), len(cols), model, feature_set
    )
    result["source"] = "magnitude_shape_functional"
    result["fit_time_s"] = time.perf_counter() - fit_started
    result["zero_target_weeks_from_2024"] = ",".join(map(str, zero_weeks.tolist()))
    return result, predictions


def evaluate_variants(
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
    log_phase("fit fixed functional hurdle models", started)
    return pd.DataFrame(results).sort_values("rmse"), predictions


def evaluate_sparse_refinements(
    variants: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    disease_2024: pd.DataFrame,
    disease_2025: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame], pd.DataFrame]:
    """Use grouped 2024 stability selection to compact the functional blocks."""
    stage_train, stage_test = add_stage_week_interactions(*variants["functional_magnitude_shape"])
    selected_variants = {
        "sparse_functional_magnitude_shape_top50": variants["functional_magnitude_shape"],
        "sparse_functional_magnitude_shape_interaction_top50": variants[
            "functional_magnitude_shape_interaction"
        ],
        "stage_sparse_functional_magnitude_shape_top50": (stage_train, stage_test),
    }
    results: list[dict[str, object]] = []
    predictions: dict[tuple[str, str], pd.DataFrame] = {}
    selections: list[pd.DataFrame] = []
    started = time.perf_counter()
    for feature_set, (train_features, test_features) in selected_variants.items():
        train = current_severity.build_current_model_table(train_features, disease_2024)
        test = current_severity.build_current_model_table(test_features, disease_2025)
        logging.info("%s sparse model table: train=%d test=%d", feature_set, len(train), len(test))
        result, pred, selection = current_severity.current_hurdle_stability_topk_model(
            train,
            test,
            feature_set,
            top_k=50,
            log_positive=False,
        )
        result["source"] = "magnitude_shape_functional_sparse"
        results.append(result)
        predictions[(result["model"], feature_set)] = pred
        selections.append(selection)
        floor_result, floor_pred = apply_sparse_zero_week_floor(train, result, pred, feature_set)
        results.append(floor_result)
        predictions[(floor_result["model"], feature_set)] = floor_pred
    log_phase("fit sparse functional refinements", started)
    selection_df = pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()
    return pd.DataFrame(results).sort_values("rmse"), predictions, selection_df


def apply_sparse_zero_week_floor(
    train: pd.DataFrame,
    result: dict[str, object],
    predictions: pd.DataFrame,
    feature_set: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    zero_weeks = (
        train.groupby("target_week")[TARGET].max().loc[lambda values: values <= 0].index.to_numpy()
    )
    model = f"{result['model']}_zero_week_floor"
    floor_pred = predictions.copy()
    floor_pred["model"] = model
    if zero_weeks.size:
        floor_pred.loc[floor_pred["target_week"].isin(zero_weeks), "y_pred"] = 0.0
    residual_pipeline.save_predictions(floor_pred, model, feature_set, COVARIATES)
    floor_result = residual_pipeline.score_predictions(
        floor_pred,
        int(result["n_train"]),
        int(result["n_features"]),
        model,
        feature_set,
    )
    floor_result["source"] = "magnitude_shape_functional_sparse"
    floor_result["fit_time_s"] = result.get("fit_time_s", math.nan)
    floor_result["classifier_features"] = result.get("classifier_features", math.nan)
    floor_result["regressor_features"] = result.get("regressor_features", math.nan)
    floor_result["feature_selection_strategy"] = result.get("feature_selection_strategy", "")
    floor_result["train_in_sample_rmse"] = result.get("train_in_sample_rmse", math.nan)
    floor_result["train_grouped_oof_rmse"] = result.get("train_grouped_oof_rmse", math.nan)
    floor_result["train_oof_minus_in_sample_rmse"] = result.get(
        "train_oof_minus_in_sample_rmse", math.nan
    )
    floor_result["external_minus_oof_rmse"] = (
        floor_result["rmse"] - float(result.get("train_grouped_oof_rmse", math.nan))
        if pd.notna(result.get("train_grouped_oof_rmse", math.nan))
        else math.nan
    )
    floor_result["zero_target_weeks_from_2024"] = ",".join(map(str, zero_weeks.tolist()))
    return floor_result, floor_pred


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
            "current_hurdle_stability_top30_raw_positive",
            "curve_embedding_fpca_only",
        ): CURVE_RESULTS_DIR
        / "predictions/severity_predictions_current_hurdle_stability_top30_raw_positive_curve_embedding_fpca_only_spectral_plus_week.csv",
    }
    out = {}
    for key, path in paths.items():
        if path.exists():
            out[key] = pd.read_csv(path)
        else:
            logging.warning("Missing context prediction: %s", path)
    return out


def score_prediction_frame(
    predictions: pd.DataFrame,
    model: str,
    feature_set: str,
    source: str,
) -> dict[str, object]:
    y = predictions["y_true"].to_numpy(float)
    pred = predictions["y_pred"].to_numpy(float)
    return {
        "model": model,
        "feature_set": feature_set,
        "source": source,
        "n_test": len(predictions),
        "n_features": math.nan,
        "rmse": math.sqrt(mean_squared_error(y, pred)),
        "mae": mean_absolute_error(y, pred),
        "r2": r2_score(y, pred) if len(np.unique(y)) > 1 else math.nan,
        "spearman": residual_pipeline.safe_spearman(y, pred),
    }


def paired_delta_vs_nadir(predictions: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    baseline_key = ("current_hurdle_top20_raw_positive", "compact_anomaly_nadir")
    if baseline_key not in predictions:
        return pd.DataFrame()
    rng = np.random.default_rng(SEED)
    rows = []
    baseline = predictions[baseline_key]
    for key, candidate in predictions.items():
        if key == baseline_key:
            continue
        model, feature_set = key
        stats = residual_pipeline.paired_bootstrap_delta_ci(baseline, candidate, rng)
        rows.append(
            {
                "model": model,
                "feature_set": feature_set,
                "baseline_model": baseline_key[0],
                "baseline_feature_set": baseline_key[1],
                "resample_unit": "plot_id",
                **stats,
            }
        )
    return pd.DataFrame(rows).sort_values("rmse_reduction_observed", ascending=False)


def write_report(
    comparison: pd.DataFrame,
    delta: pd.DataFrame,
    week_summary: pd.DataFrame,
    audit: pd.DataFrame,
    block_audit: pd.DataFrame,
    paths: dict[str, Path],
    log_path: Path,
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "magnitude_shape_functional_current_severity_summary.md"
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
        "## Results: Magnitude-Shape Functional Current Severity Model",
        "",
        "This analysis tests a fixed functional-data idea: disease severity may be encoded by the coupling between absolute reflectance magnitude and normalized angular curve shape.",
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
        markdown_table(week_summary.round(4), max_rows=40),
        "",
        "### Feature Construction Audit",
        "",
        markdown_table(audit.round(4), max_rows=30),
        "",
        "### Compact Block Embedding Audit",
        "",
        markdown_table(block_audit.round(4), max_rows=10),
        "",
        "**Interpretation**: The interaction variant is the direct test of the hypothesis: angular shape should matter most when coupled to reflectance magnitude. The sparse rows test whether that space can be made stable with 2024-only grouped feature selection, and the stage-sparse row tests whether the magnitude/shape relationship changes with known acquisition week. This is still current-severity prediction because predictor and target weeks are identical.",
        "",
        "### Reproducibility",
        "",
        "- Training year: 2024.",
        "- Test year: 2025.",
        "- Target: same-week plot-level `ds_plot`.",
        "- Inputs: VZA curves from cached multiangular reflectance distribution features.",
        "- Embedding: magnitude summaries, log-ratio angular shape FPCA/summaries, standardized within-curve magnitude-by-shape interactions, compact 2024-fitted block-PCA refinements, and a known-week stage-aware functional refinement.",
        "- Excluded predictors: treatment, cultivar, block, inoculation/design metadata, RAA, and disease history.",
        "- Models: fixed hurdle Ridge for the four main ablations; grouped 2024-only top-50 stability-selected hurdle Ridge for the sparse refinements.",
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
    train_features, test_features, audit = build_magnitude_shape_features(long_2024, long_2025)
    variants = feature_variants(train_features, test_features)
    compact_variants, block_audit = compact_block_variants(train_features, test_features)
    variants.update(compact_variants)
    functional_results, functional_predictions = evaluate_variants(
        variants, disease_2024, disease_2025
    )
    sparse_results, sparse_predictions, sparse_selection = evaluate_sparse_refinements(
        variants, disease_2024, disease_2025
    )
    functional_results = pd.concat(
        [functional_results, sparse_results],
        ignore_index=True,
        sort=False,
    ).sort_values("rmse")

    context_predictions = load_context_predictions()
    context_rows = [
        score_prediction_frame(pred, model, feature_set, "existing_context")
        for (model, feature_set), pred in context_predictions.items()
    ]
    comparison = pd.concat(
        [pd.DataFrame(context_rows), functional_results],
        ignore_index=True,
        sort=False,
    ).sort_values("rmse")
    all_predictions = {**context_predictions, **functional_predictions, **sparse_predictions}
    delta = paired_delta_vs_nadir(all_predictions)
    week_summary = current_severity.prediction_week_summary(all_predictions)

    paths = {
        "model_comparison": RESULTS_DIR / "magnitude_shape_functional_model_comparison.csv",
        "paired_delta_vs_nadir": RESULTS_DIR / "magnitude_shape_functional_delta_vs_nadir.csv",
        "target_week_summary": RESULTS_DIR / "magnitude_shape_functional_target_week_summary.csv",
        "feature_audit": RESULTS_DIR / "magnitude_shape_functional_feature_audit.csv",
        "block_embedding_audit": RESULTS_DIR
        / "magnitude_shape_functional_block_embedding_audit.csv",
        "sparse_selected_features": RESULTS_DIR
        / "magnitude_shape_functional_sparse_selected_features.csv",
        "predictions": PREDICTIONS_DIR,
    }
    comparison.to_csv(paths["model_comparison"], index=False)
    delta.to_csv(paths["paired_delta_vs_nadir"], index=False)
    week_summary.to_csv(paths["target_week_summary"], index=False)
    audit.to_csv(paths["feature_audit"], index=False)
    block_audit.to_csv(paths["block_embedding_audit"], index=False)
    sparse_selection.to_csv(paths["sparse_selected_features"], index=False)
    report_path = write_report(
        comparison,
        delta,
        week_summary,
        audit,
        block_audit,
        paths,
        log_path,
    )
    logging.info("Report: %s", report_path)
    log_phase("total", total_started)


if __name__ == "__main__":
    main()
