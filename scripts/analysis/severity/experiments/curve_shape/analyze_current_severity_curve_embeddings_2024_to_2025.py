#!/usr/bin/env python3
"""Current severity prediction from multiangular reflectance-curve embeddings."""

from __future__ import annotations

import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.severity import (
    analyze_current_plot_severity_2024_to_2025 as current_severity,
)
from scripts.analysis.severity import (
    debug_multiangular_rmse_bottleneck as residual_pipeline,
)
from scripts.analysis.severity.analyze_cross_year_generalization_2024_to_2025 import (
    SEED,
    TARGET,
    TARGET_LOG,
    WARNING_TARGET,
    WARNING_THRESHOLD,
)
from scripts.analysis.severity.analyze_multiangular_distribution_feature_family import (
    clean_token,
    markdown_table,
)

INPUT_RESULTS_DIR = ROOT / "outputs/runs/analysis/severity/future/compact_distribution_feature_family/results"
OUTPUT_ROOT = ROOT / "outputs/runs/analysis/severity/current/curve_embeddings_2024_to_2025"
RESULTS_DIR = OUTPUT_ROOT / "results"
REPORTS_DIR = OUTPUT_ROOT / "reports"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = OUTPUT_ROOT / "figures"
LOGS_DIR = ROOT / "outputs/archive/legacy_unscoped/logs"
DISEASE_2024_CLEAN = ROOT / "outputs/shared/disease/clean_disease_scores_2024.csv"
DISEASE_2025_CLEAN = ROOT / "outputs/shared/disease/clean_disease_scores_2025.csv"
CURRENT_RESULTS_DIR = ROOT / "outputs/runs/analysis/severity/current/2024_to_2025/results"

COVARIATES = "spectral_plus_week"
ANGLE_GRID = np.array([12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 52.5])
CURVE_BANDS = {"red", "red_edge", "nir", "osavi"}
REFLECTANCE_METRICS = {"mean", "p10", "p25", "iqr", "cv"}
OSAVI_METRICS = {"osavi_mean", "osavi_p10", "osavi_iqr", "osavi_frac_lt_025", "osavi_frac_lt_035"}
N_FPCA_COMPONENTS = 3


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"analyze_current_severity_curve_embeddings_2024_to_2025_{timestamp}.log"
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
    residual_pipeline.FROZEN_MANIFEST_PATH = OUTPUT_ROOT / "curve_embedding_manifest.json"


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    t0 = time.perf_counter()
    long_2024 = pd.read_csv(INPUT_RESULTS_DIR / "distribution_features_long_2024.csv")
    long_2025 = pd.read_csv(INPUT_RESULTS_DIR / "distribution_features_long_2025.csv")
    disease_2024 = pd.read_csv(DISEASE_2024_CLEAN)
    disease_2025 = pd.read_csv(DISEASE_2025_CLEAN)
    log_phase("read cached long features and disease scores", t0)
    return long_2024, long_2025, disease_2024, disease_2025


def clean_band_name(value: object) -> str:
    return clean_token(value).replace("red_edge", "red_edge")


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


def curve_matrices(
    long_2024: pd.DataFrame, long_2025: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[list[str], list[float]]]]:
    groups: dict[str, tuple[list[str], list[float]]] = {}
    frames = []
    for year, data in [(2024, filter_curve_rows(long_2024)), (2025, filter_curve_rows(long_2025))]:
        pivot = data.pivot_table(
            index=["year", "week", "plot_id", "cult", "trt", "curve_group"],
            columns="vza_midpoint",
            values="value",
            aggfunc="mean",
        ).reset_index()
        pivot.columns.name = None
        angle_cols = [
            col for col in pivot.columns if isinstance(col, float) or isinstance(col, int)
        ]
        pivot = pivot.rename(columns={angle: f"angle_{float(angle):04.1f}" for angle in angle_cols})
        frames.append((year, pivot))
    train_pivot = frames[0][1]
    test_pivot = frames[1][1]
    for group in sorted(set(train_pivot["curve_group"]).intersection(test_pivot["curve_group"])):
        cols = [
            f"angle_{angle:04.1f}"
            for angle in ANGLE_GRID
            if f"angle_{angle:04.1f}" in train_pivot.columns
        ]
        groups[group] = (cols, [float(col.replace("angle_", "")) for col in cols])
    return train_pivot, test_pivot, groups


def fit_group_embedding(
    train_group: pd.DataFrame,
    test_group: pd.DataFrame,
    group: str,
    angle_cols: list[str],
    angles: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    meta_cols = ["year", "week", "plot_id", "cult", "trt"]
    train_meta = train_group[meta_cols].reset_index(drop=True)
    test_meta = test_group[meta_cols].reset_index(drop=True)
    imputer = SimpleImputer(strategy="median")
    train_values = imputer.fit_transform(train_group[angle_cols])
    test_values = imputer.transform(test_group[angle_cols])
    angle_x = np.asarray(angles, dtype=float).reshape(-1, 1)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_values)
    test_scaled = scaler.transform(test_values)
    n_components = min(N_FPCA_COMPONENTS, train_scaled.shape[1], train_scaled.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=SEED)
    train_scores = pca.fit_transform(train_scaled)
    test_scores = pca.transform(test_scaled)

    spline = SplineTransformer(n_knots=5, degree=3, include_bias=False, extrapolation="continue")
    basis = spline.fit_transform(angle_x)
    pinv = np.linalg.pinv(basis)
    train_coef = train_values @ pinv.T
    test_coef = test_values @ pinv.T

    def shape_features(values: np.ndarray) -> dict[str, np.ndarray]:
        centered_x = np.asarray(angles, dtype=float) - float(np.mean(angles))
        denom = float(np.sum(centered_x**2))
        gradient = np.gradient(values, np.asarray(angles, dtype=float), axis=1)
        curvature = np.gradient(gradient, np.asarray(angles, dtype=float), axis=1)
        first = values[:, 0]
        last = values[:, -1]
        return {
            "auc": np.trapezoid(values, x=np.asarray(angles, dtype=float), axis=1),
            "range": np.max(values, axis=1) - np.min(values, axis=1),
            "std": np.std(values, axis=1),
            "offnadir_minus_nadir": last - first,
            "relative_offnadir_minus_nadir": np.divide(
                last - first,
                first,
                out=np.full_like(first, np.nan, dtype=float),
                where=np.isfinite(first) & (first != 0),
            ),
            "linear_slope": np.sum(
                (values - values.mean(axis=1, keepdims=True)) * centered_x, axis=1
            )
            / denom,
            "mean_abs_derivative": np.mean(np.abs(gradient), axis=1),
            "mean_abs_curvature": np.mean(np.abs(curvature), axis=1),
        }

    feature_prefix = f"curve__{group}"
    train_features = train_meta.copy()
    test_features = test_meta.copy()
    for idx in range(train_coef.shape[1]):
        train_features[f"{feature_prefix}__spline_coef_{idx:02d}"] = train_coef[:, idx]
        test_features[f"{feature_prefix}__spline_coef_{idx:02d}"] = test_coef[:, idx]
    for idx in range(n_components):
        train_features[f"{feature_prefix}__fpca_{idx + 1}"] = train_scores[:, idx]
        test_features[f"{feature_prefix}__fpca_{idx + 1}"] = test_scores[:, idx]
    for name, values in shape_features(train_values).items():
        train_features[f"{feature_prefix}__{name}"] = values
    for name, values in shape_features(test_values).items():
        test_features[f"{feature_prefix}__{name}"] = values

    audit = {
        "curve_group": group,
        "n_train_curves": len(train_group),
        "n_test_curves": len(test_group),
        "n_angles": len(angle_cols),
        "n_spline_coefficients": train_coef.shape[1],
        "n_fpca_components": n_components,
        "fpca_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
    }
    return train_features, test_features, audit


def build_curve_feature_sets(
    long_2024: pd.DataFrame, long_2025: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    started = time.perf_counter()
    train_pivot, test_pivot, groups = curve_matrices(long_2024, long_2025)
    train_frames = []
    test_frames = []
    audit_rows = []
    for group, (angle_cols, angles) in groups.items():
        if len(angle_cols) < 5:
            continue
        train_group = train_pivot.loc[train_pivot["curve_group"].eq(group)].copy()
        test_group = test_pivot.loc[test_pivot["curve_group"].eq(group)].copy()
        if train_group.empty or test_group.empty:
            continue
        train_features, test_features, audit = fit_group_embedding(
            train_group, test_group, group, angle_cols, angles
        )
        train_frames.append(train_features)
        test_frames.append(test_features)
        audit_rows.append(audit)
    if not train_frames:
        raise RuntimeError("No curve feature groups were created.")

    def merge_feature_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
        merged = frames[0]
        meta_cols = ["year", "week", "plot_id", "cult", "trt"]
        for frame in frames[1:]:
            feature_cols = [col for col in frame.columns if col not in meta_cols]
            merged = merged.merge(frame[meta_cols + feature_cols], on=meta_cols, how="outer")
        return merged

    train = merge_feature_frames(train_frames)
    test = merge_feature_frames(test_frames)
    audit = pd.DataFrame(audit_rows)
    log_phase("build curve embedding feature sets", started)
    logging.info(
        "curve embeddings: train rows=%d test rows=%d features=%d",
        len(train),
        len(test),
        len([c for c in train.columns if c not in {"year", "week", "plot_id", "cult", "trt"}]),
    )
    return train, test, audit


def build_current_model_table(features: pd.DataFrame, disease: pd.DataFrame) -> pd.DataFrame:
    disease_target = disease[["plot_id", "week", "ds_plot"]].rename(columns={"ds_plot": TARGET})
    table = features.merge(disease_target, on=["plot_id", "week"], how="inner")
    table["predictor_week"] = table["week"].astype(int)
    table["target_week"] = table["week"].astype(int)
    table[TARGET_LOG] = np.log1p(table[TARGET])
    table[WARNING_TARGET] = (table[TARGET] >= WARNING_THRESHOLD).astype(int)
    return table.sort_values(["target_week", "plot_id"]).reset_index(drop=True)


def curve_feature_variants(
    train_features: pd.DataFrame, test_features: pd.DataFrame
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    meta_cols = ["year", "week", "plot_id", "cult", "trt"]
    feature_cols = [col for col in train_features.columns if col not in meta_cols]
    variants = {
        "curve_embedding_all": feature_cols,
        "curve_embedding_fpca_shape": [col for col in feature_cols if "__spline_coef_" not in col],
        "curve_embedding_fpca_only": [col for col in feature_cols if "__fpca_" in col],
        "curve_embedding_shape_only": [
            col for col in feature_cols if "__spline_coef_" not in col and "__fpca_" not in col
        ],
    }
    out: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for name, cols in variants.items():
        out[name] = (
            train_features[meta_cols + cols].copy(),
            test_features[meta_cols + cols].copy(),
        )
        logging.info("%s features: %d", name, len(cols))
    return out


def score_predictions(
    predictions: pd.DataFrame, model: str, feature_set: str, n_features: int
) -> dict[str, object]:
    y = predictions["y_true"].to_numpy(float)
    pred = predictions["y_pred"].to_numpy(float)
    return {
        "model": model,
        "feature_set": feature_set,
        "covariates": COVARIATES,
        "n_train": math.nan,
        "n_test": len(predictions),
        "n_features": n_features,
        "rmse": math.sqrt(mean_squared_error(y, pred)),
        "mae": mean_absolute_error(y, pred),
        "r2": r2_score(y, pred),
        "spearman": residual_pipeline.safe_spearman(y, pred),
    }


def baseline_rows() -> pd.DataFrame:
    path = CURRENT_RESULTS_DIR / "current_severity_model_comparison.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    keep = df[
        (
            df["model"].isin(
                [
                    "current_hurdle_top20_raw_positive",
                    "current_hurdle_stability_top50_raw_positive",
                    "hurdle_probability_times_severity",
                ]
            )
        )
        & df["feature_set"].isin(["compact_anomaly_nadir", "compact_anomaly_multiangular"])
    ].copy()
    keep["source"] = "existing_current_severity"
    return keep


def prediction_path(model: str, feature_set: str) -> Path:
    return PREDICTIONS_DIR / (
        f"severity_predictions_{residual_pipeline.safe_filename(model)}_"
        f"{residual_pipeline.safe_filename(feature_set)}_{residual_pipeline.safe_filename(COVARIATES)}.csv"
    )


def paired_bootstrap_against_nadir(results: pd.DataFrame) -> pd.DataFrame:
    baseline_path = (
        CURRENT_RESULTS_DIR
        / "predictions/severity_predictions_current_hurdle_top20_raw_positive_compact_anomaly_nadir_spectral_plus_week.csv"
    )
    if not baseline_path.exists():
        return pd.DataFrame()
    baseline = pd.read_csv(baseline_path)
    rows = []
    rng = np.random.default_rng(SEED)
    curve_mask = results["feature_set"].astype(str).str.startswith("curve_embedding")
    for _, row in results.loc[curve_mask].iterrows():
        pred_path = prediction_path(row["model"], row["feature_set"])
        if not pred_path.exists():
            continue
        candidate = pd.read_csv(pred_path)
        stats = residual_pipeline.paired_bootstrap_delta_ci(baseline, candidate, rng)
        rows.append(
            {
                "baseline_model": "current_hurdle_top20_raw_positive",
                "baseline_feature_set": "compact_anomaly_nadir",
                "candidate_model": row["model"],
                "candidate_feature_set": row["feature_set"],
                **stats,
            }
        )
    return pd.DataFrame(rows).sort_values("rmse_reduction_observed", ascending=False)


def write_report(
    results: pd.DataFrame,
    delta: pd.DataFrame,
    audit: pd.DataFrame,
    paths: dict[str, Path],
    log_path: Path,
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "current_severity_curve_embeddings_summary.md"
    curve_mask = results["feature_set"].astype(str).str.startswith("curve_embedding")
    curve_rows = results.loc[curve_mask].copy()
    lines = [
        "## Results: Current Severity From Multiangular Curve Embeddings",
        "",
        "This analysis embeds each plot-week band/metric response as a curve over VZA using spline coefficients, FPCA scores, and derivative/shape descriptors.",
        "",
        "### Curve Model Comparison",
        "",
        markdown_table(curve_rows.round(4), max_rows=20),
        "",
        "### Context Baselines",
        "",
        markdown_table(results.loc[~curve_mask].round(4), max_rows=20),
        "",
        "### Paired Delta Versus Best Nadir",
        "",
        "Positive RMSE reduction means the curve model improved over the existing best nadir current-severity baseline.",
        "",
        markdown_table(delta.round(4), max_rows=20),
        "",
        "### Curve Embedding Audit",
        "",
        markdown_table(audit.round(4), max_rows=30),
        "",
        "### Reproducibility",
        "",
        "- Target: same-week plot-level `ds_plot`.",
        "- Training year: 2024.",
        "- Test year: 2025.",
        "- Curve domain: VZA midpoints 12.5 to 52.5 degrees.",
        "- Embeddings: B-spline coefficients, FPCA scores fit on 2024 only, and derivative/shape descriptors.",
        "- Feature variants: all curve features, FPCA+shape without spline coefficients, FPCA-only, and shape-only.",
        f"- Log: `{log_path}`",
        "",
        "### Outputs",
        "",
    ]
    lines.extend([f"- {label}: `{path}`" for label, path in paths.items()])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> None:
    total_t0 = time.perf_counter()
    configure_reused_pipeline_paths()
    for directory in [RESULTS_DIR, REPORTS_DIR, PREDICTIONS_DIR, FIGURES_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging()
    long_2024, long_2025, disease_2024, disease_2025 = read_inputs()
    train_features, test_features, audit = build_curve_feature_sets(long_2024, long_2025)
    variants = curve_feature_variants(train_features, test_features)

    results = []
    predictions = {}
    selections = []
    model_t0 = time.perf_counter()
    for feature_set, (train_features_variant, test_features_variant) in variants.items():
        train = build_current_model_table(train_features_variant, disease_2024)
        test = build_current_model_table(test_features_variant, disease_2025)
        logging.info("%s current model table: train=%d test=%d", feature_set, len(train), len(test))
        for top_k in [20, 30, 50]:
            result, pred, selection = current_severity.current_hurdle_stability_topk_model(
                train, test, feature_set, top_k=top_k, log_positive=False
            )
            results.append(result)
            predictions[(result["model"], feature_set)] = pred
            selections.append(selection)
        if feature_set in {"curve_embedding_all", "curve_embedding_fpca_shape"}:
            result, pred = residual_pipeline.fit_hurdle_model(train, test, feature_set)
            results.append(result)
            predictions[(result["model"], feature_set)] = pred
    log_phase("fit curve embedding models", model_t0)

    results_df = pd.DataFrame(results).sort_values(["rmse", "model"])
    context = baseline_rows()
    combined = pd.concat([results_df, context], ignore_index=True, sort=False)
    delta = paired_bootstrap_against_nadir(results_df)
    selection_df = pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()
    paths = {
        "model_comparison": RESULTS_DIR / "curve_embedding_model_comparison.csv",
        "context_model_comparison": RESULTS_DIR / "curve_embedding_with_context_baselines.csv",
        "paired_delta_vs_nadir": RESULTS_DIR / "curve_embedding_delta_vs_best_nadir.csv",
        "curve_embedding_audit": RESULTS_DIR / "curve_embedding_audit.csv",
        "selected_features": RESULTS_DIR
        / "curve_embedding_stability_selection_feature_frequencies.csv",
    }
    results_df.to_csv(paths["model_comparison"], index=False)
    combined.to_csv(paths["context_model_comparison"], index=False)
    delta.to_csv(paths["paired_delta_vs_nadir"], index=False)
    audit.to_csv(paths["curve_embedding_audit"], index=False)
    selection_df.to_csv(paths["selected_features"], index=False)
    report_path = write_report(combined, delta, audit, paths, log_path)
    logging.info("Report: %s", report_path)
    log_phase("total", total_t0)


if __name__ == "__main__":
    main()
