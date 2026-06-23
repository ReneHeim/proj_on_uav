"""Test the selected severity model with cultivar as an in-model covariate.

This retrains the same two-stage Ridge + residual XGBoost architecture used by
the selected compact multiangular model, but adds cultivar as a numeric
covariate. This is not a post-hoc recalibration of saved predictions.
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_cross_year_generalization_2024_to_2025 import (  # noqa: E402
    build_model_table,
    load_2024_disease_with_fallback,
    load_2025_disease_with_fallback,
)
from scripts.debug_multiangular_rmse_bottleneck import (  # noqa: E402
    COVARIATES,
    fit_tuned_xgboost_residual_with_cols,
    load_cached_features,
    prepare_aligned,
)
from scripts.test_extra_compact_features_residual_pipeline import classify_candidates  # noqa: E402

OUTPUT_ROOT = ROOT / "outputs/multiangular_distribution_feature_family/model_bottleneck_debug"
RESULTS_DIR = OUTPUT_ROOT / "results"
REPORTS_DIR = OUTPUT_ROOT / "reports"
LOGS_DIR = ROOT / "outputs/logs"

FEATURE_SET = "compact_anomaly_multiangular"
SELECTED_REFLECTANCE_FEATURES = 42
ORIGINAL_SELECTED_RESULTS = RESULTS_DIR / "selected_42_feature_severity_result.csv"


def setup_logging() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"test_selected_severity_model_with_cultivar_covariate_{timestamp}.log"
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


def load_train_test_tables():
    t0 = time.perf_counter()
    features = load_cached_features()
    disease_2024 = load_2024_disease_with_fallback()
    disease_2025, _audit = load_2025_disease_with_fallback()
    train = build_model_table(features[FEATURE_SET][0], disease_2024)
    test = build_model_table(features[FEATURE_SET][1], disease_2025)
    log_phase("load selected feature family and disease targets", t0)
    return train, test


def add_cultivar_covariate(
    train_aligned,
    test_aligned,
    train_original,
    test_original,
    cols: list[str],
) -> tuple[object, object, list[str]]:
    train = train_aligned.copy()
    test = test_aligned.copy()
    if "cult" not in train.columns:
        train_meta = train_original[["plot_id", "cult"]].drop_duplicates()
        train = train.merge(train_meta, on="plot_id", how="left")
    if "cult" not in test.columns:
        test_meta = test_original[["plot_id", "cult"]].drop_duplicates()
        test = test.merge(test_meta, on="plot_id", how="left")
    if train["cult"].isna().any() or test["cult"].isna().any():
        raise RuntimeError("Could not map cultivar metadata back to aligned train/test rows.")
    train["known__cultivar_capone"] = (train["cult"].astype(str).str.lower() == "capone").astype(float)
    test["known__cultivar_capone"] = (test["cult"].astype(str).str.lower() == "capone").astype(float)
    return train, test, cols + ["known__cultivar_capone"]


def main() -> None:
    log_path = setup_logging()
    started = time.perf_counter()
    train, test = load_train_test_tables()

    # Use the same accepted-first feature order used to identify the selected
    # 42-feature result. This ensures the only test change is adding cultivar.
    candidates, timing_cols, ranked_reflectance, train_aligned, test_aligned = classify_candidates(train, test)
    selected_cols = timing_cols + ranked_reflectance[:SELECTED_REFLECTANCE_FEATURES]
    train_cult, test_cult, selected_cols_with_cultivar = add_cultivar_covariate(
        train_aligned,
        test_aligned,
        train,
        test,
        selected_cols,
    )

    result, predictions, tuning = fit_tuned_xgboost_residual_with_cols(
        train_cult,
        test_cult,
        selected_cols_with_cultivar,
        "selected_42_plus_cultivar_covariate_residual_xgboost",
        FEATURE_SET,
    )
    original = pl.read_csv(ORIGINAL_SELECTED_RESULTS).row(0, named=True)

    result_row = {
        "model": "selected_42_plus_cultivar_covariate_residual_xgboost",
        "feature_set": FEATURE_SET,
        "n_reflectance_features": SELECTED_REFLECTANCE_FEATURES,
        "n_timing_features": len(timing_cols),
        "n_cultivar_features": 1,
        "n_total_features": len(selected_cols_with_cultivar),
        "rmse": result["rmse"],
        "mae": result["mae"],
        "r2": result["r2"],
        "spearman": result["spearman"],
        "original_selected42_rmse": float(original["rmse"]),
        "original_selected42_mae": float(original["mae"]),
        "original_selected42_r2": float(original["r2"]),
        "original_selected42_spearman": float(original["spearman"]),
        "rmse_change_vs_original_selected42": result["rmse"] - float(original["rmse"]),
        "mae_change_vs_original_selected42": result["mae"] - float(original["mae"]),
        "delta_r2_vs_original_selected42": result["r2"] - float(original["r2"]),
        "delta_spearman_vs_original_selected42": result["spearman"] - float(original["spearman"]),
        "xgboost_config": result.get("xgboost_config"),
        "best_iteration": result.get("best_iteration"),
        "eval_rmse_2024": result.get("eval_rmse_2024"),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / "selected42_plus_cultivar_covariate_result.csv"
    tuning_path = RESULTS_DIR / "selected42_plus_cultivar_covariate_tuning.csv"
    prediction_path = RESULTS_DIR / "selected42_plus_cultivar_covariate_predictions.csv"
    pl.DataFrame([result_row]).write_csv(result_path)
    pl.from_pandas(tuning).write_csv(tuning_path)
    pl.from_pandas(predictions).write_csv(prediction_path)

    report_path = REPORTS_DIR / "selected42_plus_cultivar_covariate_summary.md"
    report_path.write_text(build_report(result_row, result_path, tuning_path, prediction_path, log_path), encoding="utf-8")
    logging.info("Result: %s", result_path)
    logging.info("Report: %s", report_path)
    log_phase("total runtime", started)


def build_report(row: dict, result_path: Path, tuning_path: Path, prediction_path: Path, log_path: Path) -> str:
    direction = "worse" if row["rmse_change_vs_original_selected42"] > 0 else "better"
    return f"""## Results: Selected Severity Model + Cultivar Covariate

This test retrains the original selected two-stage severity model with cultivar included as an in-model covariate. It is not a post-hoc recalibration.

| Model | Total features | RMSE | MAE | R2 | Spearman |
|---|---:|---:|---:|---:|---:|
| Selected 42-feature model | 44 | {row['original_selected42_rmse']:.3f} | {row['original_selected42_mae']:.3f} | {row['original_selected42_r2']:.3f} | {row['original_selected42_spearman']:.3f} |
| Selected 42 + cultivar covariate | {row['n_total_features']} | {row['rmse']:.3f} | {row['mae']:.3f} | {row['r2']:.3f} | {row['spearman']:.3f} |

| Change after adding cultivar | Value |
|---|---:|
| RMSE change | {row['rmse_change_vs_original_selected42']:.3f} |
| MAE change | {row['mae_change_vs_original_selected42']:.3f} |
| Delta R2 | {row['delta_r2_vs_original_selected42']:.3f} |
| Delta Spearman | {row['delta_spearman_vs_original_selected42']:.3f} |

**Interpretation**: Adding cultivar as a model covariate made RMSE {direction} by {abs(row['rmse_change_vs_original_selected42']):.3f} severity units relative to the selected 42-feature model.

**Outputs**:

- Result CSV: `{result_path.relative_to(ROOT)}`
- Tuning CSV: `{tuning_path.relative_to(ROOT)}`
- Prediction CSV: `{prediction_path.relative_to(ROOT)}`
- Log: `{log_path.relative_to(ROOT)}`
"""


if __name__ == "__main__":
    main()
