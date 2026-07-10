#!/usr/bin/env python3
"""Current severity from hurdle ElasticNet and hurdle additive GAM curve models."""

from __future__ import annotations

from src.research.common import write_report as persist_report

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
from sklearn.linear_model import ElasticNetCV, LassoCV, LogisticRegressionCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import SplineTransformer, StandardScaler

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
from scripts.analysis.severity.experiments.curve_shape.analyze_current_severity_curve_only_elasticnet_gam_2024_to_2025 import (
    build_curve_summary_features,
    fit_gam_transformer,
    select_top_correlated_features,
    transform_gam,
)
from scripts.analysis.severity.experiments.curve_shape.analyze_current_severity_curve_only_functional_2024_to_2025 import (
    build_sampled_curve_features,
    make_curve_sources,
)
from scripts.analysis.severity.experiments.curve_shape.analyze_current_severity_magnitude_shape_functional_2024_to_2025 import (
    COVARIATES,
    paired_delta_vs_nadir,
    read_inputs,
    score_prediction_frame,
)
from scripts.analysis.severity.experiments.curve_shape.analyze_current_severity_sparse_functional_discriminant_shape_2024_to_2025 import (
    RAA_2024,
    RAA_2025,
    read_raa,
)
from scripts.analysis.severity.analyze_multiangular_distribution_feature_family import (
    markdown_table,
)

OUTPUT_ROOT = ROOT / "outputs/runs/analysis/severity/current/curve_only_hurdle_elasticnet_gam_2024_to_2025"
RESULTS_DIR = OUTPUT_ROOT / "results"
REPORTS_DIR = OUTPUT_ROOT / "reports"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = OUTPUT_ROOT / "figures"
LOGS_DIR = ROOT / "outputs/archive/legacy_unscoped/logs"

CURRENT_RESULTS_DIR = ROOT / "outputs/runs/analysis/severity/current/2024_to_2025/results"
CURVE_ONLY_RESULTS_DIR = (
    ROOT / "outputs/runs/analysis/severity/current/curve_only_functional_2024_to_2025/results"
)
DIRECT_CURVE_RESULTS_DIR = (
    ROOT / "outputs/runs/analysis/severity/current/curve_only_elasticnet_gam_2024_to_2025/results"
)

TARGET = current_severity.TARGET
WARNING_TARGET = current_severity.WARNING_TARGET
SEED = current_severity.SEED
EPS = 1e-8
SMOOTH_RIDGE_ALPHAS = np.logspace(-1, 3, 5)
SMOOTH_RIDGE_LAMBDAS = np.array([0.0, 1.0, 10.0, 100.0, 1000.0])


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = (
        LOGS_DIR
        / f"analyze_current_severity_curve_only_hurdle_elasticnet_gam_2024_to_2025_{timestamp}.log"
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
        OUTPUT_ROOT / "curve_only_hurdle_elasticnet_gam_manifest.json"
    )


def regression_scores(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": math.sqrt(mean_squared_error(y_true, pred)),
        "mae": mean_absolute_error(y_true, pred),
        "r2": r2_score(y_true, pred) if len(np.unique(y_true)) > 1 else math.nan,
        "spearman": residual_pipeline.safe_spearman(y_true, pred),
    }


def cv_splits(groups: np.ndarray, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    return list(
        GroupKFold(n_splits=min(5, len(np.unique(groups)))).split(np.zeros(len(y)), y, groups)
    )


def classifier_cv_folds(y_present: np.ndarray) -> int | None:
    counts = np.bincount(y_present.astype(int), minlength=2)
    smallest_class = int(counts.min())
    if smallest_class < 2:
        return None
    return min(3, smallest_class)


def fit_elasticnet_regressor(
    x: pd.DataFrame, y: np.ndarray, groups: np.ndarray
) -> tuple[ElasticNetCV, SimpleImputer, StandardScaler]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(imputer.fit_transform(x))
    model = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9, 1.0],
        alphas=np.logspace(-2, 2, 35),
        cv=cv_splits(groups, y),
        max_iter=50000,
        random_state=SEED,
    )
    model.fit(x_scaled, y)
    return model, imputer, scaler


def fit_lasso_regressor(
    x: pd.DataFrame, y: np.ndarray, groups: np.ndarray
) -> tuple[LassoCV, SimpleImputer, StandardScaler]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(imputer.fit_transform(x))
    model = LassoCV(
        alphas=np.logspace(-3, 2, 50),
        cv=cv_splits(groups, y),
        max_iter=50000,
        random_state=SEED,
    )
    model.fit(x_scaled, y)
    return model, imputer, scaler


def smooth_difference_matrix(cols: list[str]) -> np.ndarray:
    grouped: dict[str, list[tuple[float, int]]] = {}
    for idx, col in enumerate(cols):
        if "__vza_" not in col:
            continue
        prefix, angle_text = col.rsplit("__vza_", 1)
        try:
            angle = float(angle_text)
        except ValueError:
            continue
        grouped.setdefault(prefix, []).append((angle, idx))

    rows: list[np.ndarray] = []
    for values in grouped.values():
        ordered = sorted(values)
        for (_, left_idx), (_, right_idx) in zip(ordered[:-1], ordered[1:]):
            row = np.zeros(len(cols), dtype=float)
            row[right_idx] = 1.0
            row[left_idx] = -1.0
            rows.append(row)
    if not rows:
        return np.zeros((0, len(cols)), dtype=float)
    return np.vstack(rows)


def solve_smooth_ridge(
    x: np.ndarray, y_centered: np.ndarray, dmat: np.ndarray, alpha: float, smooth_lambda: float
) -> np.ndarray:
    penalty = alpha * np.eye(x.shape[1])
    if dmat.size:
        penalty = penalty + smooth_lambda * (dmat.T @ dmat)
    system = x.T @ x + penalty
    rhs = x.T @ y_centered
    return np.linalg.solve(system, rhs)


def fit_smooth_ridge_regressor(
    x: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[dict[str, object], SimpleImputer, StandardScaler]:
    cols = list(x.columns)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(imputer.fit_transform(x))
    y_mean = float(np.mean(y))
    y_centered = y - y_mean
    dmat = smooth_difference_matrix(cols)

    best: dict[str, object] | None = None
    for alpha in SMOOTH_RIDGE_ALPHAS:
        for smooth_lambda in SMOOTH_RIDGE_LAMBDAS:
            fold_rmse: list[float] = []
            for fit_idx, eval_idx in cv_splits(groups, y):
                coef = solve_smooth_ridge(
                    x_scaled[fit_idx],
                    y_centered[fit_idx],
                    dmat,
                    float(alpha),
                    float(smooth_lambda),
                )
                pred = x_scaled[eval_idx] @ coef + float(np.mean(y[fit_idx]))
                fold_rmse.append(math.sqrt(mean_squared_error(y[eval_idx], pred)))
            mean_rmse = float(np.mean(fold_rmse))
            if best is None or mean_rmse < float(best["cv_rmse"]):
                best = {
                    "alpha": float(alpha),
                    "smooth_lambda": float(smooth_lambda),
                    "cv_rmse": mean_rmse,
                }
    if best is None:
        raise RuntimeError("Smooth ridge model selection failed.")
    coef = solve_smooth_ridge(
        x_scaled,
        y_centered,
        dmat,
        float(best["alpha"]),
        float(best["smooth_lambda"]),
    )
    model = {
        "coef": coef,
        "intercept": y_mean,
        "alpha": best["alpha"],
        "smooth_lambda": best["smooth_lambda"],
        "cv_rmse": best["cv_rmse"],
        "n_difference_penalties": int(dmat.shape[0]),
        "coef_roughness": float(np.mean((dmat @ coef) ** 2)) if dmat.size else 0.0,
    }
    return model, imputer, scaler


def predict_elasticnet(
    model: ElasticNetCV, imputer: SimpleImputer, scaler: StandardScaler, x: pd.DataFrame
) -> np.ndarray:
    return model.predict(scaler.transform(imputer.transform(x)))


def predict_lasso(
    model: LassoCV, imputer: SimpleImputer, scaler: StandardScaler, x: pd.DataFrame
) -> np.ndarray:
    return model.predict(scaler.transform(imputer.transform(x)))


def predict_smooth_ridge(
    model: dict[str, object], imputer: SimpleImputer, scaler: StandardScaler, x: pd.DataFrame
) -> np.ndarray:
    coef = np.asarray(model["coef"], dtype=float)
    return scaler.transform(imputer.transform(x)) @ coef + float(model["intercept"])


def fit_hurdle_sparse_linear_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[str],
    regressor_kind: str,
) -> tuple[np.ndarray, dict[str, object]]:
    y = train[TARGET].to_numpy(float)
    groups = train["plot_id"].to_numpy()
    y_present = (y > 0).astype(int)
    imputer_c = SimpleImputer(strategy="median")
    scaler_c = StandardScaler()
    x_class = scaler_c.fit_transform(imputer_c.fit_transform(train[cols]))
    x_class_test = scaler_c.transform(imputer_c.transform(test[cols]))
    class_cv = classifier_cv_folds(y_present)
    if len(np.unique(y_present)) < 2 or class_cv is None:
        prob = np.full(len(test), float(y_present.mean()))
    else:
        classifier = LogisticRegressionCV(
            Cs=np.logspace(-2, 1, 8),
            cv=class_cv,
            class_weight="balanced",
            solver="liblinear",
            max_iter=5000,
            random_state=SEED,
        )
        classifier.fit(x_class, y_present)
        prob = classifier.predict_proba(x_class_test)[:, 1]
    pos = y > 0
    if pos.sum() < 5:
        severity = np.full(len(test), float(np.nanmean(y)))
        audit = {}
    else:
        if regressor_kind == "elasticnet":
            reg, imputer_r, scaler_r = fit_elasticnet_regressor(
                train.loc[pos, cols],
                y[pos],
                train.loc[pos, "plot_id"].to_numpy(),
            )
            severity = predict_elasticnet(reg, imputer_r, scaler_r, test[cols])
            audit = {
                "elasticnet_alpha": float(reg.alpha_),
                "elasticnet_l1_ratio": float(reg.l1_ratio_),
                "n_nonzero_coefficients": int(np.count_nonzero(np.abs(reg.coef_) > EPS)),
            }
        elif regressor_kind == "lasso":
            reg, imputer_r, scaler_r = fit_lasso_regressor(
                train.loc[pos, cols],
                y[pos],
                train.loc[pos, "plot_id"].to_numpy(),
            )
            severity = predict_lasso(reg, imputer_r, scaler_r, test[cols])
            audit = {
                "lasso_alpha": float(reg.alpha_),
                "n_nonzero_coefficients": int(np.count_nonzero(np.abs(reg.coef_) > EPS)),
            }
        elif regressor_kind == "smooth_ridge":
            reg, imputer_r, scaler_r = fit_smooth_ridge_regressor(
                train.loc[pos, cols],
                y[pos],
                train.loc[pos, "plot_id"].to_numpy(),
            )
            severity = predict_smooth_ridge(reg, imputer_r, scaler_r, test[cols])
            audit = {
                "smooth_ridge_alpha": float(reg["alpha"]),
                "smooth_ridge_lambda": float(reg["smooth_lambda"]),
                "smooth_ridge_cv_rmse": float(reg["cv_rmse"]),
                "smooth_ridge_coef_roughness": float(reg["coef_roughness"]),
                "n_difference_penalties": int(reg["n_difference_penalties"]),
                "n_nonzero_coefficients": len(cols),
            }
        else:
            raise ValueError(regressor_kind)
    return residual_pipeline.clip_predictions(prob * severity, y), audit


def fit_hurdle_elasticnet_predict(
    train: pd.DataFrame, test: pd.DataFrame, cols: list[str]
) -> tuple[np.ndarray, dict[str, object]]:
    return fit_hurdle_sparse_linear_predict(train, test, cols, "elasticnet")


def fit_hurdle_lasso_predict(
    train: pd.DataFrame, test: pd.DataFrame, cols: list[str]
) -> tuple[np.ndarray, dict[str, object]]:
    return fit_hurdle_sparse_linear_predict(train, test, cols, "lasso")


def fit_hurdle_smooth_ridge_predict(
    train: pd.DataFrame, test: pd.DataFrame, cols: list[str]
) -> tuple[np.ndarray, dict[str, object]]:
    return fit_hurdle_sparse_linear_predict(train, test, cols, "smooth_ridge")


def fit_hurdle_gam_predict(
    train: pd.DataFrame, test: pd.DataFrame, cols: list[str]
) -> tuple[np.ndarray, dict[str, object]]:
    y = train[TARGET].to_numpy(float)
    y_present = (y > 0).astype(int)
    selected_class = select_top_correlated_features(train[cols], y_present.astype(float), top_k=35)
    x_class_train, x_class_test, class_transformer = fit_gam_transformer(
        train[selected_class], test[selected_class]
    )
    class_cv = classifier_cv_folds(y_present)
    if len(np.unique(y_present)) < 2 or class_cv is None:
        prob = np.full(len(test), float(y_present.mean()))
    else:
        classifier = LogisticRegressionCV(
            Cs=np.logspace(-2, 1, 8),
            cv=class_cv,
            class_weight="balanced",
            solver="liblinear",
            max_iter=5000,
            random_state=SEED,
        )
        classifier.fit(x_class_train, y_present)
        prob = classifier.predict_proba(x_class_test)[:, 1]
    pos = y > 0
    if pos.sum() < 5:
        severity = np.full(len(test), float(np.nanmean(y)))
        audit = {}
    else:
        selected_reg = select_top_correlated_features(train.loc[pos, cols], y[pos], top_k=35)
        x_reg_train, x_reg_test, reg_transformer = fit_gam_transformer(
            train.loc[pos, selected_reg], test[selected_reg]
        )
        reg = RidgeCV(alphas=np.logspace(-2, 4, 40), cv=3)
        reg.fit(x_reg_train, y[pos])
        severity = reg.predict(x_reg_test)
        audit = {
            "ridge_alpha": float(reg.alpha_),
            "selected_classifier_summary_features": len(selected_class),
            "selected_regressor_summary_features": len(selected_reg),
            "classifier_basis_terms": len(class_transformer.expanded_names),
            "regressor_basis_terms": len(reg_transformer.expanded_names),
        }
    return residual_pipeline.clip_predictions(prob * severity, y), audit


def grouped_oof(train: pd.DataFrame, cols: list[str], model_kind: str) -> np.ndarray:
    y = train[TARGET].to_numpy(float)
    groups = train["plot_id"].to_numpy()
    pred = np.zeros(len(train), dtype=float)
    for fit_idx, eval_idx in cv_splits(groups, y):
        if model_kind == "elasticnet":
            fold_pred, _ = fit_hurdle_elasticnet_predict(
                train.iloc[fit_idx], train.iloc[eval_idx], cols
            )
        elif model_kind == "lasso":
            fold_pred, _ = fit_hurdle_lasso_predict(train.iloc[fit_idx], train.iloc[eval_idx], cols)
        elif model_kind == "smooth_ridge":
            fold_pred, _ = fit_hurdle_smooth_ridge_predict(
                train.iloc[fit_idx], train.iloc[eval_idx], cols
            )
        elif model_kind == "gam":
            fold_pred, _ = fit_hurdle_gam_predict(train.iloc[fit_idx], train.iloc[eval_idx], cols)
        else:
            raise ValueError(model_kind)
        pred[eval_idx] = fold_pred
    return pred


def evaluate_model(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    disease_2024: pd.DataFrame,
    disease_2025: pd.DataFrame,
    feature_set: str,
    model_kind: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    started = time.perf_counter()
    train = current_severity.build_current_model_table(train_features, disease_2024)
    test = current_severity.build_current_model_table(test_features, disease_2025)
    cols, train_aligned, test_aligned = residual_pipeline.prepare_aligned(train, test)
    if model_kind == "elasticnet":
        pred, audit = fit_hurdle_elasticnet_predict(train_aligned, test_aligned, cols)
        model_name = "hurdle_elasticnet_curve_regression"
        source = "curve_only_hurdle_elasticnet"
    elif model_kind == "lasso":
        pred, audit = fit_hurdle_lasso_predict(train_aligned, test_aligned, cols)
        model_name = "hurdle_lasso_curve_regression"
        source = "curve_only_hurdle_lasso"
    elif model_kind == "smooth_ridge":
        pred, audit = fit_hurdle_smooth_ridge_predict(train_aligned, test_aligned, cols)
        model_name = "hurdle_smooth_ridge_curve_regression"
        source = "curve_only_hurdle_smooth_ridge"
    else:
        pred, audit = fit_hurdle_gam_predict(train_aligned, test_aligned, cols)
        model_name = "hurdle_sparse_additive_gam"
        source = "curve_summary_hurdle_gam"
    in_sample_pred, _ = (
        fit_hurdle_elasticnet_predict(train_aligned, train_aligned, cols)
        if model_kind == "elasticnet"
        else (
            fit_hurdle_lasso_predict(train_aligned, train_aligned, cols)
            if model_kind == "lasso"
            else (
                fit_hurdle_smooth_ridge_predict(train_aligned, train_aligned, cols)
                if model_kind == "smooth_ridge"
                else fit_hurdle_gam_predict(train_aligned, train_aligned, cols)
            )
        )
    )
    oof_pred = grouped_oof(train_aligned, cols, model_kind)
    predictions = residual_pipeline.prediction_frame(test_aligned, pred, model_name, feature_set)
    residual_pipeline.save_predictions(predictions, model_name, feature_set, COVARIATES)
    result = residual_pipeline.score_predictions(
        predictions, len(train_aligned), len(cols), model_name, feature_set
    )
    y = train_aligned[TARGET].to_numpy(float)
    in_scores = regression_scores(y, in_sample_pred)
    oof_scores = regression_scores(y, oof_pred)
    result["source"] = source
    result["fit_time_s"] = time.perf_counter() - started
    result["train_in_sample_rmse"] = in_scores["rmse"]
    result["train_grouped_oof_rmse"] = oof_scores["rmse"]
    result["train_oof_minus_in_sample_rmse"] = oof_scores["rmse"] - in_scores["rmse"]
    result["external_minus_oof_rmse"] = result["rmse"] - oof_scores["rmse"]
    result.update(audit)
    return result, predictions


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
        (
            "direct_elasticnet_curve_regression",
            "direct_elasticnet_vza_log_curve_samples",
        ): DIRECT_CURVE_RESULTS_DIR
        / "predictions/severity_predictions_direct_elasticnet_curve_regression_direct_elasticnet_vza_log_curve_samples_spectral_plus_week.csv",
        (
            "sparse_additive_spline_gam",
            "sparse_gam_vza_log_curve_summaries",
        ): DIRECT_CURVE_RESULTS_DIR
        / "predictions/severity_predictions_sparse_additive_spline_gam_sparse_gam_vza_log_curve_summaries_spectral_plus_week.csv",
    }
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for key, path in paths.items():
        if path.exists():
            out[key] = pd.read_csv(path)
        else:
            logging.warning("Missing context prediction: %s", path)
    return out


def build_report(
    comparison: pd.DataFrame, delta: pd.DataFrame, paths: dict[str, Path], log_path: Path
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "curve_only_hurdle_elasticnet_gam_current_severity_summary.md"
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
        "lasso_alpha",
        "smooth_ridge_alpha",
        "smooth_ridge_lambda",
        "smooth_ridge_cv_rmse",
        "smooth_ridge_coef_roughness",
        "n_difference_penalties",
        "n_nonzero_coefficients",
        "ridge_alpha",
    ]
    display = comparison.copy()
    for col in display_cols:
        if col not in display.columns:
            display[col] = math.nan
    lines = [
        "## Results: Hurdle Lasso, ElasticNet, and GAM Current Severity",
        "",
        "This analysis tests whether LassoCV, ElasticNet, smooth-ridge regularization, and additive GAM improve once given the same zero/nonzero hurdle structure as the winning curve-only Ridge model.",
        "",
        "### Model Comparison",
        "",
        markdown_table(display[display_cols].round(4).sort_values("rmse"), max_rows=20),
        "",
        "### Paired Delta Versus Existing Nadir",
        "",
        markdown_table(delta.round(4), max_rows=20),
        "",
        "**Interpretation**: The hurdle layer substantially rescues sparse linear models relative to direct ElasticNet, but these sparse alternatives still do not beat the existing selected curve-only hurdle Ridge. Smooth-ridge regularization tests whether enforcing adjacent VZA coefficient smoothness reduces feature overfitting; if it closes the external-minus-OOF gap, the failure was unstable isolated curve-bin weights rather than the hurdle structure.",
        "",
        "### Reproducibility",
        "",
        "- Training year: 2024.",
        "- Test year: 2025.",
        "- Target: same-week plot-level `ds_plot`.",
        "- Inputs: VZA log-reflectance curve samples for hurdle Lasso, ElasticNet, and smooth Ridge; VZA log-curve summaries for hurdle GAM.",
        "- Validation: grouped 2024 OOF by plot and external 2025 test.",
        f"- Log: `{log_path}`",
        "",
        "### Outputs",
        "",
    ]
    lines.extend([f"- {label}: `{path}`" for label, path in paths.items()])
    persist_report(report_path, lines)
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
    summary_train, summary_test, _ = build_curve_summary_features(train_sources, test_sources)

    en_result, en_pred = evaluate_model(
        sampled_train,
        sampled_test,
        disease_2024,
        disease_2025,
        "hurdle_elasticnet_vza_log_curve_samples",
        "elasticnet",
    )
    lasso_result, lasso_pred = evaluate_model(
        sampled_train,
        sampled_test,
        disease_2024,
        disease_2025,
        "hurdle_lasso_vza_log_curve_samples",
        "lasso",
    )
    smooth_result, smooth_pred = evaluate_model(
        sampled_train,
        sampled_test,
        disease_2024,
        disease_2025,
        "hurdle_smooth_ridge_vza_log_curve_samples",
        "smooth_ridge",
    )
    gam_result, gam_pred = evaluate_model(
        summary_train,
        summary_test,
        disease_2024,
        disease_2025,
        "hurdle_gam_vza_log_curve_summaries",
        "gam",
    )
    context_predictions = load_context_predictions()
    context_rows = [
        score_prediction_frame(pred, model, feature_set, "existing_context")
        for (model, feature_set), pred in context_predictions.items()
    ]
    new_predictions = {
        (en_result["model"], en_result["feature_set"]): en_pred,
        (lasso_result["model"], lasso_result["feature_set"]): lasso_pred,
        (smooth_result["model"], smooth_result["feature_set"]): smooth_pred,
        (gam_result["model"], gam_result["feature_set"]): gam_pred,
    }
    comparison = pd.concat(
        [
            pd.DataFrame(context_rows),
            pd.DataFrame([en_result, lasso_result, smooth_result, gam_result]),
        ],
        ignore_index=True,
        sort=False,
    ).sort_values("rmse")
    delta = paired_delta_vs_nadir({**context_predictions, **new_predictions})
    paths = {
        "model_comparison": RESULTS_DIR / "curve_only_hurdle_elasticnet_gam_model_comparison.csv",
        "paired_delta_vs_nadir": RESULTS_DIR
        / "curve_only_hurdle_elasticnet_gam_delta_vs_nadir.csv",
        "predictions": PREDICTIONS_DIR,
    }
    comparison.to_csv(paths["model_comparison"], index=False)
    delta.to_csv(paths["paired_delta_vs_nadir"], index=False)
    report_path = build_report(comparison, delta, paths, log_path)
    logging.info("Report: %s", report_path)
    log_phase("total", total_started)


if __name__ == "__main__":
    main()
