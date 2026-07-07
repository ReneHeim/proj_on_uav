#!/usr/bin/env python3
"""Current severity from healthy counterfactual reflectance residuals.

Hypothesis: angular curve shape describes canopy geometry. Disease severity is
then the amount by which observed spectral magnitude is weaker than expected
for a healthy canopy with the same angular structure.

The first stage is trained only on healthy/low-disease 2024 plot-weeks:

    angular shape + heterogeneity -> expected healthy magnitude

The second stage predicts current severity from:

    observed magnitude - expected healthy magnitude

No treatment, cultivar, block, inoculation, RAA, or disease-history predictors
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
from sklearn.linear_model import RidgeCV
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
from scripts.analysis.severity.analyze_current_severity_curve_embeddings_2024_to_2025 import (
    ANGLE_GRID,
    INPUT_RESULTS_DIR,
    clean_band_name,
)
from scripts.analysis.severity.analyze_multiangular_distribution_feature_family import (
    clean_token,
    markdown_table,
)

OUTPUT_ROOT = ROOT / "outputs/current_severity_healthy_counterfactual_2024_to_2025"
RESULTS_DIR = OUTPUT_ROOT / "results"
REPORTS_DIR = OUTPUT_ROOT / "reports"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = OUTPUT_ROOT / "figures"
LOGS_DIR = ROOT / "outputs/logs"

DISEASE_2024_CLEAN = ROOT / "outputs/disease/clean_disease_scores_2024.csv"
DISEASE_2025_CLEAN = ROOT / "outputs/disease/clean_disease_scores_2025.csv"
CURRENT_RESULTS_DIR = ROOT / "outputs/current_severity_2024_to_2025/results"
FUNCTIONAL_RESULTS_DIR = (
    ROOT / "outputs/current_severity_magnitude_shape_functional_2024_to_2025/results"
)

COVARIATES = "spectral_plus_week"
META_COLS = ["year", "week", "plot_id", "cult", "trt"]
TARGET = current_severity.TARGET
TARGET_LOG = current_severity.TARGET_LOG
WARNING_TARGET = current_severity.WARNING_TARGET
WARNING_THRESHOLD = current_severity.WARNING_THRESHOLD
SEED = current_severity.SEED
ALPHAS = current_severity.ALPHAS
EPS = 1e-4
HEALTHY_MAX_SEVERITY_FOR_COUNTERFACTUAL = 0.5
COUNTERFACTUAL_PCA_VARIANCE = 0.90
COUNTERFACTUAL_MAX_PCS = 12

TARGET_MAGNITUDES = [
    ("nir", "mean"),
    ("red_edge", "mean"),
    ("red", "mean"),
    ("osavi", "osavi_mean"),
    ("nir", "p10"),
    ("red_edge", "p10"),
    ("red", "p10"),
    ("osavi", "osavi_p10"),
]


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"analyze_current_severity_healthy_counterfactual_2024_to_2025_{timestamp}.log"
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
    residual_pipeline.FROZEN_MANIFEST_PATH = OUTPUT_ROOT / "healthy_counterfactual_manifest.json"


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    started = time.perf_counter()
    long_2024 = pd.read_csv(INPUT_RESULTS_DIR / "distribution_features_long_2024.csv")
    long_2025 = pd.read_csv(INPUT_RESULTS_DIR / "distribution_features_long_2025.csv")
    disease_2024 = pd.read_csv(DISEASE_2024_CLEAN)
    disease_2025 = pd.read_csv(DISEASE_2025_CLEAN)
    log_phase("read cached long features and disease scores", started)
    return long_2024, long_2025, disease_2024, disease_2025


def filter_rows(long: pd.DataFrame) -> pd.DataFrame:
    data = long.copy()
    data["band_token"] = data["band_name"].map(clean_band_name)
    data["metric_token"] = data["metric"].map(clean_token)
    reflectance_keep = data["band_token"].isin(["red", "red_edge", "nir"]) & data[
        "metric_token"
    ].isin(["mean", "p10", "p25", "iqr", "cv"])
    osavi_keep = data["band_token"].eq("osavi") & data["metric_token"].isin(
        ["osavi_mean", "osavi_p10", "osavi_iqr", "osavi_frac_lt_025", "osavi_frac_lt_035"]
    )
    return data.loc[reflectance_keep | osavi_keep].copy()


def pivot_curves(long: pd.DataFrame) -> pd.DataFrame:
    data = filter_rows(long)
    pivot = data.pivot_table(
        index=META_COLS + ["band_token", "metric_token"],
        columns="vza_midpoint",
        values="value",
        aggfunc="mean",
    ).reset_index()
    pivot.columns.name = None
    angle_cols = [col for col in pivot.columns if isinstance(col, (int, float))]
    return pivot.rename(columns={angle: f"angle_{float(angle):04.1f}" for angle in angle_cols})


def row_curve(row: pd.Series) -> np.ndarray | None:
    cols = [f"angle_{angle:04.1f}" for angle in ANGLE_GRID if f"angle_{angle:04.1f}" in row.index]
    cols = [col for col in cols if pd.notna(row[col])]
    if len(cols) < 5:
        return None
    return row[cols].to_numpy(float)


def curves_for_plot_week(pivot: pd.DataFrame) -> dict[tuple, dict[tuple[str, str], np.ndarray]]:
    out: dict[tuple, dict[tuple[str, str], np.ndarray]] = {}
    for _, row in pivot.iterrows():
        key = tuple(row[col] for col in META_COLS)
        values = row_curve(row)
        if values is None:
            continue
        out.setdefault(key, {})[(str(row["band_token"]), str(row["metric_token"]))] = values
    return out


def curve_shape(values: np.ndarray) -> np.ndarray:
    shift = max(EPS, EPS - float(np.nanmin(values)))
    log_values = np.log(np.clip(values + shift, EPS, None))
    return log_values - log_values[0]


def curve_slope(values: np.ndarray) -> float:
    x = ANGLE_GRID[: len(values)]
    x = x - float(np.mean(x))
    denom = float(np.sum(x**2))
    return float(np.sum((values - values.mean()) * x) / denom) if denom > 0 else 0.0


def summarize_curve(values: np.ndarray, prefix: str) -> dict[str, float]:
    low = values[:3]
    high = values[-3:]
    return {
        f"{prefix}__mean": float(np.mean(values)),
        f"{prefix}__std": float(np.std(values)),
        f"{prefix}__range": float(np.max(values) - np.min(values)),
        f"{prefix}__high_minus_low": float(np.mean(high) - np.mean(low)),
        f"{prefix}__slope": curve_slope(values),
        f"{prefix}__roughness": float(np.mean(np.abs(np.diff(values)))),
    }


def build_structure_and_observed_features(
    curve_map: dict[tuple, dict[tuple[str, str], np.ndarray]]
) -> pd.DataFrame:
    rows = []
    for key, curves in curve_map.items():
        meta = dict(zip(META_COLS, key, strict=False))
        row: dict[str, object] = dict(meta)
        for (band, metric), values in curves.items():
            token = f"{band}__{metric}"
            row[f"observed__{token}__nadir"] = float(values[0])
            row[f"observed__{token}__mean"] = float(np.mean(values))
            row[f"observed__{token}__high_angle_mean"] = float(np.mean(values[-3:]))
            shape = curve_shape(values)
            row.update(summarize_curve(shape, f"shape__{token}"))
            if metric in {"iqr", "cv", "osavi_iqr", "osavi_frac_lt_025", "osavi_frac_lt_035"}:
                row.update(summarize_curve(values, f"heterogeneity__{token}"))

        for band in ["red", "red_edge", "nir"]:
            mean_curve = curves.get((band, "mean"))
            for quantile in ["p10", "p25"]:
                q_curve = curves.get((band, quantile))
                if mean_curve is None or q_curve is None:
                    continue
                gap = mean_curve - q_curve
                rel_gap = np.divide(gap, np.maximum(np.abs(mean_curve), EPS))
                row.update(summarize_curve(rel_gap, f"relative_weak_tail__{band}__mean_minus_{quantile}"))
        mean_osavi = curves.get(("osavi", "osavi_mean"))
        p10_osavi = curves.get(("osavi", "osavi_p10"))
        if mean_osavi is not None and p10_osavi is not None:
            row.update(summarize_curve(mean_osavi - p10_osavi, "weak_tail__osavi__mean_minus_p10"))
        rows.append(row)
    return pd.DataFrame(rows)


def target_column(band: str, metric: str) -> str:
    return f"observed__{band}__{metric}__mean"


def structure_columns(frame: pd.DataFrame) -> list[str]:
    blocked_prefixes = ("observed__", "expected__", "counterfactual__")
    return [
        col
        for col in frame.columns
        if col not in META_COLS and not col.startswith(blocked_prefixes)
    ]


def counterfactual_design_matrices(
    train_base: pd.DataFrame,
    test_base: pd.DataFrame,
    x_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_values = scaler.fit_transform(imputer.fit_transform(train_base[x_cols]))
    test_values = scaler.transform(imputer.transform(test_base[x_cols]))
    max_components = min(COUNTERFACTUAL_MAX_PCS, train_values.shape[1], train_values.shape[0] - 1)
    pca_full = PCA(n_components=max_components, random_state=SEED)
    pca_full.fit(train_values)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative, COUNTERFACTUAL_PCA_VARIANCE) + 1)
    n_components = max(1, min(n_components, max_components))
    pca = PCA(n_components=n_components, random_state=SEED)
    train_scores = pca.fit_transform(train_values)
    test_scores = pca.transform(test_values)
    train_design = pd.DataFrame(
        {f"structure_pc_{idx + 1}": train_scores[:, idx] for idx in range(n_components)},
        index=train_base.index,
    )
    test_design = pd.DataFrame(
        {f"structure_pc_{idx + 1}": test_scores[:, idx] for idx in range(n_components)},
        index=test_base.index,
    )
    week_mean = float(train_base["week"].mean())
    week_std = float(train_base["week"].std())
    if week_std <= EPS:
        week_std = 1.0
    for design, base in [(train_design, train_base), (test_design, test_base)]:
        week_z = (base["week"].to_numpy(float) - week_mean) / week_std
        design["phenology_week_z"] = week_z
        design["phenology_week_z2"] = week_z**2
    audit = {
        "raw_structure_features": float(len(x_cols)),
        "counterfactual_structure_pcs": float(n_components),
        "counterfactual_pca_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
    }
    return train_design, test_design, audit


def fit_counterfactual_features(
    train_base: pd.DataFrame,
    test_base: pd.DataFrame,
    disease_2024: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    started = time.perf_counter()
    x_cols = structure_columns(train_base)
    train_design, test_design, design_audit = counterfactual_design_matrices(
        train_base, test_base, x_cols
    )
    train = train_base.merge(disease_2024[["plot_id", "week", "ds_plot"]], on=["plot_id", "week"], how="left")
    train["ds_plot"] = train["ds_plot"].fillna(0.0)
    healthy_mask = train["ds_plot"].le(HEALTHY_MAX_SEVERITY_FOR_COUNTERFACTUAL)
    train_residual_rows = train_base[META_COLS].copy()
    test_residual_rows = test_base[META_COLS].copy()
    audit_rows = []

    for band, metric in TARGET_MAGNITUDES:
        y_col = target_column(band, metric)
        if y_col not in train.columns or y_col not in test_base.columns:
            continue
        fit_mask = healthy_mask & train[y_col].notna()
        if int(fit_mask.sum()) < 20:
            continue
        model = Pipeline(
            [
                ("ridge", RidgeCV(alphas=ALPHAS)),
            ]
        )
        model.fit(train_design.loc[fit_mask], train.loc[fit_mask, y_col].to_numpy(float))
        expected_train = model.predict(train_design)
        expected_test = model.predict(test_design)
        observed_train = train[y_col].to_numpy(float)
        observed_test = test_base[y_col].to_numpy(float)
        residual_train = observed_train - expected_train
        residual_test = observed_test - expected_test
        healthy_residual = residual_train[fit_mask.to_numpy()]
        residual_scale = float(np.nanstd(healthy_residual))
        if residual_scale <= EPS:
            residual_scale = 1.0
        token = f"{band}__{metric}"
        for out, observed, expected, residual in [
            (train_residual_rows, observed_train, expected_train, residual_train),
            (test_residual_rows, observed_test, expected_test, residual_test),
        ]:
            out[f"counterfactual__{token}__observed"] = observed
            out[f"counterfactual__{token}__expected_healthy"] = expected
            out[f"counterfactual__{token}__residual"] = residual
            out[f"counterfactual__{token}__relative_residual"] = np.divide(
                residual,
                np.maximum(np.abs(expected), EPS),
            )
            out[f"counterfactual__{token}__z_residual"] = residual / residual_scale
        healthy_pred = model.predict(train_design.loc[fit_mask])
        healthy_y = train.loc[fit_mask, y_col].to_numpy(float)
        audit_rows.append(
            {
                "target": token,
                "n_healthy_fit_rows": int(fit_mask.sum()),
                **design_audit,
                "healthy_fit_rmse": math.sqrt(mean_squared_error(healthy_y, healthy_pred)),
                "healthy_fit_r2": r2_score(healthy_y, healthy_pred)
                if len(np.unique(healthy_y)) > 1
                else math.nan,
                "healthy_residual_std": residual_scale,
            }
        )
    log_phase("fit healthy counterfactual magnitude models", started)
    return train_residual_rows, test_residual_rows, pd.DataFrame(audit_rows)


def build_feature_sets(
    long_2024: pd.DataFrame,
    long_2025: pd.DataFrame,
    disease_2024: pd.DataFrame,
) -> tuple[dict[str, tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame, pd.DataFrame]:
    started = time.perf_counter()
    train_base = build_structure_and_observed_features(curves_for_plot_week(pivot_curves(long_2024)))
    test_base = build_structure_and_observed_features(curves_for_plot_week(pivot_curves(long_2025)))
    train_counter, test_counter, counter_audit = fit_counterfactual_features(
        train_base, test_base, disease_2024
    )
    residual_cols = [
        col
        for col in train_counter.columns
        if col.startswith("counterfactual__")
        and ("__residual" in col or "__relative_residual" in col or "__z_residual" in col)
    ]
    expected_cols = [
        col
        for col in train_counter.columns
        if col.startswith("counterfactual__") and "__expected_healthy" in col
    ]
    struct_cols = structure_columns(train_base)
    feature_sets = {
        "healthy_counterfactual_residuals": (
            train_counter[META_COLS + residual_cols].copy(),
            test_counter[META_COLS + residual_cols].copy(),
        ),
        "healthy_counterfactual_residuals_expected": (
            train_counter[META_COLS + residual_cols + expected_cols].copy(),
            test_counter[META_COLS + residual_cols + expected_cols].copy(),
        ),
        "healthy_counterfactual_residuals_structure": (
            train_counter[META_COLS + residual_cols].merge(
                train_base[META_COLS + struct_cols], on=META_COLS, how="left"
            ),
            test_counter[META_COLS + residual_cols].merge(
                test_base[META_COLS + struct_cols], on=META_COLS, how="left"
            ),
        ),
    }
    support_rows = []
    for name, (train, test) in feature_sets.items():
        support_rows.append(
            {
                "feature_set": name,
                "train_rows": len(train),
                "test_rows": len(test),
                "train_features": len([col for col in train.columns if col not in META_COLS]),
                "test_features": len([col for col in test.columns if col not in META_COLS]),
            }
        )
        logging.info("%s: train=%d test=%d features=%d", name, len(train), len(test), support_rows[-1]["train_features"])
    log_phase("build healthy counterfactual feature sets", started)
    return feature_sets, counter_audit, pd.DataFrame(support_rows)


def apply_zero_week_floor(
    train: pd.DataFrame,
    result: dict[str, object],
    predictions: pd.DataFrame,
    feature_set: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    zero_weeks = (
        train.groupby("target_week")[TARGET]
        .max()
        .loc[lambda values: values <= 0]
        .index.to_numpy()
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
    for key in [
        "fit_time_s",
        "classifier_features",
        "regressor_features",
        "feature_selection_strategy",
        "train_in_sample_rmse",
        "train_grouped_oof_rmse",
        "train_oof_minus_in_sample_rmse",
    ]:
        floor_result[key] = result.get(key, math.nan)
    floor_result["external_minus_oof_rmse"] = (
        floor_result["rmse"] - float(result.get("train_grouped_oof_rmse", math.nan))
        if pd.notna(result.get("train_grouped_oof_rmse", math.nan))
        else math.nan
    )
    floor_result["source"] = "healthy_counterfactual"
    floor_result["zero_target_weeks_from_2024"] = ",".join(map(str, zero_weeks.tolist()))
    return floor_result, floor_pred


def evaluate_feature_sets(
    feature_sets: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    disease_2024: pd.DataFrame,
    disease_2025: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame], pd.DataFrame]:
    results = []
    predictions: dict[tuple[str, str], pd.DataFrame] = {}
    selections = []
    started = time.perf_counter()
    for feature_set, (train_features, test_features) in feature_sets.items():
        train = current_severity.build_current_model_table(train_features, disease_2024)
        test = current_severity.build_current_model_table(test_features, disease_2025)
        logging.info("%s current model table: train=%d test=%d", feature_set, len(train), len(test))
        result, pred, selection = current_severity.current_hurdle_stability_topk_model(
            train,
            test,
            feature_set,
            top_k=50,
            log_positive=False,
        )
        result["source"] = "healthy_counterfactual"
        results.append(result)
        predictions[(result["model"], feature_set)] = pred
        selections.append(selection)
        floor_result, floor_pred = apply_zero_week_floor(train, result, pred, feature_set)
        results.append(floor_result)
        predictions[(floor_result["model"], feature_set)] = floor_pred
    log_phase("fit healthy counterfactual severity models", started)
    selection_df = pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()
    return pd.DataFrame(results).sort_values("rmse"), predictions, selection_df


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
            "current_hurdle_stability_top50_raw_positive_zero_week_floor",
            "sparse_functional_magnitude_shape_top50",
        ): FUNCTIONAL_RESULTS_DIR
        / "predictions/severity_predictions_current_hurdle_stability_top50_raw_positive_zero_week_floor_sparse_functional_magnitude_shape_top50_spectral_plus_week.csv",
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
    counter_audit: pd.DataFrame,
    support: pd.DataFrame,
    selections: pd.DataFrame,
    paths: dict[str, Path],
    log_path: Path,
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "healthy_counterfactual_current_severity_summary.md"
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
        "train_grouped_oof_rmse",
        "external_minus_oof_rmse",
    ]
    selected = selections[selections["selected_for_final_model"]].copy() if not selections.empty else pd.DataFrame()
    selected_summary = (
        selected.groupby(["feature_set", "role"], as_index=False)
        .size()
        .rename(columns={"size": "n_selected"})
        if not selected.empty
        else pd.DataFrame()
    )
    lines = [
        "## Results: Healthy Counterfactual Current Severity Model",
        "",
        "This analysis estimates the healthy reflectance magnitude expected from angular canopy structure, then uses observed-minus-expected residuals as disease signal.",
        "",
        "### Model Comparison",
        "",
        markdown_table(comparison[display_cols].round(4).sort_values("rmse"), max_rows=20),
        "",
        "### Paired Delta Versus Existing Nadir",
        "",
        markdown_table(delta.round(4), max_rows=20),
        "",
        "### Target-Week Diagnostics",
        "",
        markdown_table(week_summary.round(4), max_rows=50),
        "",
        "### Healthy Counterfactual Fit Audit",
        "",
        markdown_table(counter_audit.round(4), max_rows=20),
        "",
        "### Feature Support",
        "",
        markdown_table(support.round(4), max_rows=10),
        "",
        "### Selected Feature Count",
        "",
        markdown_table(selected_summary, max_rows=20),
        "",
        "**Interpretation**: This is current-severity prediction, not early warning. The counterfactual stage is fitted only on healthy/low-disease 2024 plot-weeks and transferred unchanged to 2025.",
        "",
        "### Reproducibility",
        "",
        "- Training year: 2024.",
        "- Test year: 2025.",
        "- Healthy counterfactual fit rows: 2024 plot-weeks with `ds_plot <= 0.5`.",
        "- Inputs: VZA-binned reflectance distribution curves.",
        "- Excluded predictors: treatment, cultivar, block, inoculation/design metadata, RAA, and disease history.",
        "- Model: grouped 2024-only top-50 stability-selected hurdle Ridge, plus current-severity zero-week floor rows.",
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
    feature_sets, counter_audit, support = build_feature_sets(long_2024, long_2025, disease_2024)
    results, predictions, selections = evaluate_feature_sets(feature_sets, disease_2024, disease_2025)
    context_predictions = load_context_predictions()
    context_rows = [
        score_prediction_frame(pred, model, feature_set, "existing_context")
        for (model, feature_set), pred in context_predictions.items()
    ]
    comparison = pd.concat(
        [pd.DataFrame(context_rows), results],
        ignore_index=True,
        sort=False,
    ).sort_values("rmse")
    all_predictions = {**context_predictions, **predictions}
    delta = paired_delta_vs_nadir(all_predictions)
    week_summary = current_severity.prediction_week_summary(all_predictions)
    paths = {
        "model_comparison": RESULTS_DIR / "healthy_counterfactual_model_comparison.csv",
        "paired_delta_vs_nadir": RESULTS_DIR / "healthy_counterfactual_delta_vs_nadir.csv",
        "target_week_summary": RESULTS_DIR / "healthy_counterfactual_target_week_summary.csv",
        "counterfactual_fit_audit": RESULTS_DIR / "healthy_counterfactual_fit_audit.csv",
        "feature_support": RESULTS_DIR / "healthy_counterfactual_feature_support.csv",
        "selected_features": RESULTS_DIR / "healthy_counterfactual_selected_features.csv",
        "predictions": PREDICTIONS_DIR,
    }
    comparison.to_csv(paths["model_comparison"], index=False)
    delta.to_csv(paths["paired_delta_vs_nadir"], index=False)
    week_summary.to_csv(paths["target_week_summary"], index=False)
    counter_audit.to_csv(paths["counterfactual_fit_audit"], index=False)
    support.to_csv(paths["feature_support"], index=False)
    selections.to_csv(paths["selected_features"], index=False)
    report_path = write_report(
        comparison,
        delta,
        week_summary,
        counter_audit,
        support,
        selections,
        paths,
        log_path,
    )
    logging.info("Report: %s", report_path)
    log_phase("total", total_started)


if __name__ == "__main__":
    main()
