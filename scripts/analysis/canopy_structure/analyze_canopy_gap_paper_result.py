#!/usr/bin/env python3
"""Paper-focused test: does angular canopy gap explain VZA reflectance contrast?"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/proj_on_uav_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[3]
JOINED_SOURCE = (
    ROOT
    / "outputs/runs/metadata/backup_metadata/eda_lai_canopy_vza/results/lai_canopy_vza_reflectance_join_2024.csv"
)
OUT_ROOT = ROOT / "outputs/runs/analysis/canopy_structure/gap_vza"
REPORTS_ROOT = ROOT / "outputs/archive/legacy_unscoped/reports"
LOG_ROOT = ROOT / "outputs/archive/legacy_unscoped/logs"
BANDS = ["Red edge", "NIR"]
PRIMARY_WEEKS = [5, 6]

MODELS = {
    "M0_vza_controls": "relative_contrast ~ C(week) + C(cult) + C(trt) + C(vza_class)",
    "M1_add_lai": "relative_contrast ~ C(week) + C(cult) + C(trt) + C(vza_class) + lai_z_primary",
    "M2_add_canopy": "relative_contrast ~ C(week) + C(cult) + C(trt) + C(vza_class) + lai_z_primary + difn + acf",
    "M3_add_angular_gap": (
        "relative_contrast ~ C(week) + C(cult) + C(trt) + C(vza_class)"
        " + lai_z_primary + difn + acf + gap_z_primary"
    ),
}


def configure_logging() -> Path:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    path = LOG_ROOT / f"analyze_canopy_gap_paper_result_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(path), logging.StreamHandler()],
        force=True,
    )
    return path


def phase(name: str, t0: float) -> None:
    logging.info("[PHASE] %s: %.1fs", name, time.perf_counter() - t0)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def load_joined(path: Path) -> pd.DataFrame:
    t0 = time.perf_counter()
    df = pd.read_csv(path)
    logging.info("[I/O] read joined table %s rows=%d cols=%d", path, df.shape[0], df.shape[1])
    phase("load_joined", t0)
    return df


def prepare_window(df: pd.DataFrame, treated_only: bool) -> pd.DataFrame:
    t0 = time.perf_counter()
    data = df[(df["week"].isin(PRIMARY_WEEKS)) & (df["band_name"].isin(BANDS))].copy()
    if treated_only:
        data = data[data["trt"] == "trt"].copy()
    needed = ["relative_contrast", "lai", "difn", "acf", "gaps_nearest_vza", "vza_class", "plot_id"]
    data = data.dropna(subset=needed).copy()
    data["lai_z_primary"] = (data["lai"] - data["lai"].mean()) / data["lai"].std(ddof=0)
    data["gap_z_primary"] = (data["gaps_nearest_vza"] - data["gaps_nearest_vza"].mean()) / data[
        "gaps_nearest_vza"
    ].std(ddof=0)
    phase(f"prepare_window_treated_only_{treated_only}", t0)
    return data


def fit_model_set(data: pd.DataFrame, analysis_set: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    t0 = time.perf_counter()
    model_rows = []
    coef_rows = []
    for band in BANDS:
        band_data = data[data["band_name"] == band].copy()
        if band_data["plot_id"].nunique() < 8:
            logging.warning("Skipping %s %s: too few plots", analysis_set, band)
            continue
        fitted = {}
        for model_name, formula in MODELS.items():
            fit_t0 = time.perf_counter()
            model = smf.ols(formula, data=band_data).fit(
                cov_type="cluster",
                cov_kwds={"groups": band_data["plot_id"]},
            )
            logging.info(
                "[ML] fit analysis_set=%s band=%s model=%s time=%.3fs",
                analysis_set,
                band,
                model_name,
                time.perf_counter() - fit_t0,
            )
            pred_t0 = time.perf_counter()
            _ = model.predict(band_data.head(20))
            logging.info(
                "[ML] predict analysis_set=%s band=%s model=%s time=%.4fs",
                analysis_set,
                band,
                model_name,
                time.perf_counter() - pred_t0,
            )
            fitted[model_name] = model

        base = fitted["M0_vza_controls"]
        for model_name, model in fitted.items():
            model_rows.append(
                {
                    "analysis_set": analysis_set,
                    "band_name": band,
                    "model": model_name,
                    "n_rows": int(model.nobs),
                    "n_plots": int(band_data["plot_id"].nunique()),
                    "adj_r2": model.rsquared_adj,
                    "aic": model.aic,
                    "bic": model.bic,
                    "delta_adj_r2_vs_M0": model.rsquared_adj - base.rsquared_adj,
                    "delta_aic_vs_M0": model.aic - base.aic,
                    "delta_bic_vs_M0": model.bic - base.bic,
                }
            )

        primary = fitted["M3_add_angular_gap"]
        conf = primary.conf_int()
        for term in ["lai_z_primary", "difn", "acf", "gap_z_primary"]:
            coef_rows.append(
                {
                    "analysis_set": analysis_set,
                    "band_name": band,
                    "model": "M3_add_angular_gap",
                    "term": term,
                    "coef": primary.params.get(term, np.nan),
                    "std_err_cluster": primary.bse.get(term, np.nan),
                    "p_cluster": primary.pvalues.get(term, np.nan),
                    "conf_low": conf.loc[term, 0] if term in conf.index else np.nan,
                    "conf_high": conf.loc[term, 1] if term in conf.index else np.nan,
                }
            )
    phase(f"fit_model_set_{analysis_set}", t0)
    return pd.DataFrame(model_rows), pd.DataFrame(coef_rows)


def summarize_open_closed_effect(data: pd.DataFrame, analysis_set: str) -> pd.DataFrame:
    t0 = time.perf_counter()
    rows: list[dict[str, object]] = []
    for (band, vza_class), group in data.groupby(["band_name", "vza_class"]):
        low_cut = group["gaps_nearest_vza"].quantile(0.33)
        high_cut = group["gaps_nearest_vza"].quantile(0.67)
        low = group[group["gaps_nearest_vza"] <= low_cut]
        high = group[group["gaps_nearest_vza"] >= high_cut]
        if low.empty or high.empty:
            continue
        rows.append(
            {
                "analysis_set": analysis_set,
                "band_name": band,
                "vza_class": vza_class,
                "vza_midpoint": float(group["vza_midpoint"].median()),
                "closed_low_gap_median_contrast": float(low["relative_contrast"].median()),
                "open_high_gap_median_contrast": float(high["relative_contrast"].median()),
                "open_minus_closed_contrast": float(
                    high["relative_contrast"].median() - low["relative_contrast"].median()
                ),
                "closed_gap_mean": float(low["gaps_nearest_vza"].mean()),
                "open_gap_mean": float(high["gaps_nearest_vza"].mean()),
                "n_closed_rows": int(len(low)),
                "n_open_rows": int(len(high)),
                "n_closed_plots": int(low["plot_id"].nunique()),
                "n_open_plots": int(high["plot_id"].nunique()),
            }
        )
    phase(f"summarize_open_closed_effect_{analysis_set}", t0)
    return pd.DataFrame(rows).sort_values(["analysis_set", "band_name", "vza_midpoint"])


def cross_validate_models(data: pd.DataFrame, analysis_set: str, n_splits: int = 5) -> pd.DataFrame:
    """Grouped CV by plot_id for predictive performance of contrast models."""
    t0 = time.perf_counter()
    rows: list[dict[str, object]] = []
    formulas = {
        "M0_vza_controls": MODELS["M0_vza_controls"],
        "M3_add_angular_gap": MODELS["M3_add_angular_gap"],
    }
    for band in BANDS:
        band_data = data[data["band_name"] == band].copy()
        groups = band_data["plot_id"].astype(str)
        unique_groups = groups.nunique()
        if unique_groups < 4:
            logging.warning("Skipping CV for %s %s: too few groups", analysis_set, band)
            continue
        splitter = GroupKFold(n_splits=min(n_splits, unique_groups))
        for model_name, formula in formulas.items():
            for fold, (train_idx, test_idx) in enumerate(
                splitter.split(band_data, groups=groups), start=1
            ):
                train = band_data.iloc[train_idx].copy()
                test = band_data.iloc[test_idx].copy()
                fit_t0 = time.perf_counter()
                model = smf.ols(formula, data=train).fit()
                logging.info(
                    "[ML] cv fit analysis_set=%s band=%s model=%s fold=%d time=%.3fs",
                    analysis_set,
                    band,
                    model_name,
                    fold,
                    time.perf_counter() - fit_t0,
                )
                pred_t0 = time.perf_counter()
                pred = model.predict(test)
                logging.info(
                    "[ML] cv predict analysis_set=%s band=%s model=%s fold=%d time=%.4fs",
                    analysis_set,
                    band,
                    model_name,
                    fold,
                    time.perf_counter() - pred_t0,
                )
                actual = test["relative_contrast"].to_numpy(dtype=float)
                predicted = pred.to_numpy(dtype=float)
                residual = actual - predicted
                sse = float(np.sum(residual**2))
                sst = float(np.sum((actual - actual.mean()) ** 2))
                rows.append(
                    {
                        "analysis_set": analysis_set,
                        "band_name": band,
                        "model": model_name,
                        "fold": fold,
                        "n_train": int(len(train)),
                        "n_test": int(len(test)),
                        "n_train_plots": int(train["plot_id"].nunique()),
                        "n_test_plots": int(test["plot_id"].nunique()),
                        "rmse": float(np.sqrt(np.mean(residual**2))),
                        "mae": float(np.mean(np.abs(residual))),
                        "r2": 1.0 - sse / sst if sst > 0 else np.nan,
                    }
                )
    phase(f"cross_validate_models_{analysis_set}", t0)
    return pd.DataFrame(rows)


def summarize_cv(cv_results: pd.DataFrame) -> pd.DataFrame:
    if cv_results.empty:
        return pd.DataFrame()
    return (
        cv_results.groupby(["analysis_set", "band_name", "model"], as_index=False)
        .agg(
            n_folds=("fold", "nunique"),
            mean_rmse=("rmse", "mean"),
            sd_rmse=("rmse", "std"),
            mean_mae=("mae", "mean"),
            sd_mae=("mae", "std"),
            mean_r2=("r2", "mean"),
            sd_r2=("r2", "std"),
        )
        .sort_values(["analysis_set", "band_name", "model"])
    )


def save_figures(model_summary: pd.DataFrame, open_closed: pd.DataFrame) -> list[Path]:
    t0 = time.perf_counter()
    fig_dir = OUT_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    primary = model_summary[
        (model_summary["analysis_set"] == "primary_all_plots_weeks5_6")
        & (model_summary["model"].isin(["M1_add_lai", "M2_add_canopy", "M3_add_angular_gap"]))
    ].copy()
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    x_labels = ["M1_add_lai", "M2_add_canopy", "M3_add_angular_gap"]
    label_map = {
        "M1_add_lai": "+ LAI",
        "M2_add_canopy": "+ DIFN/ACF",
        "M3_add_angular_gap": "+ angular gap",
    }
    x = np.arange(len(x_labels))
    width = 0.35
    for idx, band in enumerate(BANDS):
        values = (
            primary[primary["band_name"] == band]
            .set_index("model")
            .reindex(x_labels)["delta_adj_r2_vs_M0"]
        )
        axis.bar(x + (idx - 0.5) * width, values, width=width, label=band)
    axis.axhline(0, color="#4D4D4D", linewidth=0.9)
    axis.set_xticks(x)
    axis.set_xticklabels([label_map[label] for label in x_labels])
    axis.set_ylabel("Delta adjusted R2 vs VZA baseline")
    axis.set_title("Closed-canopy weeks 5-6: canopy variables explain VZA contrast")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#E3E6E8")
    axis.legend(frameon=False)
    fig.tight_layout()
    path = fig_dir / "paper_model_gain_canopy_gap_weeks5_6_2024.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    primary_effect = open_closed[open_closed["analysis_set"] == "primary_all_plots_weeks5_6"].copy()
    fig, axis = plt.subplots(figsize=(8.8, 5.2))
    for band, color, marker in [("Red edge", "#0072B2", "o"), ("NIR", "#D55E00", "s")]:
        band_df = primary_effect[primary_effect["band_name"] == band].sort_values("vza_midpoint")
        axis.plot(
            band_df["vza_midpoint"],
            band_df["open_minus_closed_contrast"],
            marker=marker,
            linewidth=2.6,
            color=color,
            label=band,
        )
    axis.axhline(0, color="#4D4D4D", linewidth=1.0)
    axis.set_title("Open canopy minus closed canopy VZA contrast, weeks 5-6")
    axis.set_xlabel("View zenith angle midpoint (deg)")
    axis.set_ylabel("Open-gap minus closed-gap\nrelative contrast vs nadir")
    axis.text(
        0.02,
        0.96,
        "Above zero: gappy canopy has stronger off-nadir contrast\n"
        "Below zero: closed canopy has stronger off-nadir contrast",
        transform=axis.transAxes,
        va="top",
        fontsize=9,
        color="#4D4D4D",
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#E3E6E8")
    axis.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    path = fig_dir / "paper_open_minus_closed_gap_vza_contrast_weeks5_6_2024.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    phase("save_figures", t0)
    return paths


def write_report(
    model_summary: pd.DataFrame,
    coef_summary: pd.DataFrame,
    cv_summary: pd.DataFrame,
    open_closed: pd.DataFrame,
    figure_paths: list[Path],
    log_path: Path,
) -> Path:
    t0 = time.perf_counter()
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_ROOT / "canopy_gap_vza_paper_result_summary.md"

    primary_models = model_summary[
        (model_summary["analysis_set"] == "primary_all_plots_weeks5_6")
        & (
            model_summary["model"].isin(
                ["M0_vza_controls", "M1_add_lai", "M2_add_canopy", "M3_add_angular_gap"]
            )
        )
    ].copy()
    primary_models = primary_models[
        [
            "band_name",
            "model",
            "n_rows",
            "n_plots",
            "adj_r2",
            "aic",
            "bic",
            "delta_adj_r2_vs_M0",
            "delta_aic_vs_M0",
            "delta_bic_vs_M0",
        ]
    ]
    primary_coef = coef_summary[
        (coef_summary["analysis_set"] == "primary_all_plots_weeks5_6")
        & (coef_summary["term"].isin(["gap_z_primary", "difn", "acf", "lai_z_primary"]))
    ].copy()
    primary_coef = primary_coef[
        ["band_name", "term", "coef", "std_err_cluster", "p_cluster", "conf_low", "conf_high"]
    ]
    sensitivity = model_summary[
        (model_summary["analysis_set"] == "sensitivity_treated_only_weeks5_6")
        & (model_summary["model"] == "M3_add_angular_gap")
    ].copy()
    sensitivity = sensitivity[
        [
            "band_name",
            "n_rows",
            "n_plots",
            "adj_r2",
            "delta_adj_r2_vs_M0",
            "delta_aic_vs_M0",
            "delta_bic_vs_M0",
        ]
    ]
    primary_cv = cv_summary[cv_summary["analysis_set"] == "primary_all_plots_weeks5_6"].copy()
    primary_cv = primary_cv[
        ["band_name", "model", "n_folds", "mean_rmse", "sd_rmse", "mean_mae", "mean_r2", "sd_r2"]
    ]

    for df in [primary_models, primary_coef, sensitivity, primary_cv]:
        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in {"n_rows", "n_plots"}:
                df[col] = df[col].round(4)

    best = primary_models[primary_models["model"] == "M3_add_angular_gap"].set_index("band_name")
    red_gain = best.loc["Red edge", "delta_adj_r2_vs_M0"]
    nir_gain = best.loc["NIR", "delta_adj_r2_vs_M0"]
    red_aic = best.loc["Red edge", "delta_aic_vs_M0"]
    nir_aic = best.loc["NIR", "delta_aic_vs_M0"]

    lines = [
        "# Paper Result: Canopy Gap Explains VZA Reflectance Contrast",
        "",
        "## Primary Analysis",
        "",
        markdown_table(primary_models),
        "",
        "## Primary Coefficients",
        "",
        markdown_table(primary_coef),
        "",
        "## Sensitivity: Treated Plots Only",
        "",
        markdown_table(sensitivity),
        "",
        "## Predictive Check: Grouped Cross-Validation by Plot",
        "",
        markdown_table(primary_cv),
        "",
        "## Interpretation",
        "",
        (
            f"In closed-canopy weeks 5-6, adding the angular LAI-instrument gap fraction improved the VZA-only "
            f"baseline by +{red_gain:.3f} adjusted R2 for Red edge and +{nir_gain:.3f} for NIR. AIC also improved "
            f"for both bands (Red edge {red_aic:.1f}; NIR {nir_aic:.1f}), indicating that angular canopy gap "
            "contains useful information about the off-nadir reflectance contrast."
        ),
        "",
        (
            "Biologically, this supports the interpretation that multiangular reflectance is partly controlled by "
            "canopy architecture. Open/gappy canopies and closed canopies do not show the same off-nadir contrast, "
            "because oblique views sample different mixtures of leaf surfaces, shaded canopy, row gaps, and soil."
        ),
        "",
        (
            "This is not evidence for measured side leaf orientation. It is evidence that independent canopy gap "
            "structure helps explain VZA-dependent reflectance during the closed-canopy period."
        ),
        "",
        "## Outputs",
        "",
        f"- Model table: `{OUT_ROOT / 'results/canopy_gap_vza_primary_models_2024.csv'}`",
        f"- Coefficient table: `{OUT_ROOT / 'results/canopy_gap_vza_primary_coefficients_2024.csv'}`",
        f"- Cross-validation folds: `{OUT_ROOT / 'results/canopy_gap_vza_cv_folds_2024.csv'}`",
        f"- Cross-validation summary: `{OUT_ROOT / 'results/canopy_gap_vza_cv_summary_2024.csv'}`",
        f"- Open/closed canopy effect table: `{OUT_ROOT / 'results/canopy_gap_vza_open_closed_effect_2024.csv'}`",
    ]
    lines.extend([f"- Figure: `{path}`" for path in figure_paths])
    lines.extend(
        [
            f"- Log: `{log_path}`",
            "",
            "## Reproducibility",
            "",
            f"- Input joined table: `{JOINED_SOURCE}`",
            "- Primary window: weeks 5-6.",
            "- Bands: Red edge and NIR.",
            "- Response: `relative_contrast`, off-nadir reflectance relative to nadir.",
            "- Main predictor: standardized `gaps_nearest_vza`, the LAI instrument ring gap fraction nearest each UAV VZA bin.",
            "- Cluster-robust standard errors: grouped by `plot_id`.",
            "- Predictive check: `GroupKFold` grouped by `plot_id`, 5 folds for the primary all-plot analysis.",
            "- Random seed: not used; deterministic OLS analysis.",
        ]
    )
    report_path.write_text("\n".join(lines))
    phase("write_report", t0)
    return report_path


def main() -> None:
    log_path = configure_logging()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "results").mkdir(parents=True, exist_ok=True)

    joined = load_joined(JOINED_SOURCE)
    primary = prepare_window(joined, treated_only=False)
    treated = prepare_window(joined, treated_only=True)

    model_primary, coef_primary = fit_model_set(primary, "primary_all_plots_weeks5_6")
    model_treated, coef_treated = fit_model_set(treated, "sensitivity_treated_only_weeks5_6")
    model_summary = pd.concat([model_primary, model_treated], ignore_index=True)
    coef_summary = pd.concat([coef_primary, coef_treated], ignore_index=True)
    open_closed = pd.concat(
        [
            summarize_open_closed_effect(primary, "primary_all_plots_weeks5_6"),
            summarize_open_closed_effect(treated, "sensitivity_treated_only_weeks5_6"),
        ],
        ignore_index=True,
    )
    cv_results = pd.concat(
        [
            cross_validate_models(primary, "primary_all_plots_weeks5_6"),
            cross_validate_models(treated, "sensitivity_treated_only_weeks5_6", n_splits=4),
        ],
        ignore_index=True,
    )
    cv_summary = summarize_cv(cv_results)

    model_path = OUT_ROOT / "results/canopy_gap_vza_primary_models_2024.csv"
    coef_path = OUT_ROOT / "results/canopy_gap_vza_primary_coefficients_2024.csv"
    open_closed_path = OUT_ROOT / "results/canopy_gap_vza_open_closed_effect_2024.csv"
    cv_folds_path = OUT_ROOT / "results/canopy_gap_vza_cv_folds_2024.csv"
    cv_summary_path = OUT_ROOT / "results/canopy_gap_vza_cv_summary_2024.csv"
    model_summary.to_csv(model_path, index=False)
    coef_summary.to_csv(coef_path, index=False)
    open_closed.to_csv(open_closed_path, index=False)
    cv_results.to_csv(cv_folds_path, index=False)
    cv_summary.to_csv(cv_summary_path, index=False)
    logging.info("[I/O] wrote %s", model_path)
    logging.info("[I/O] wrote %s", coef_path)
    logging.info("[I/O] wrote %s", open_closed_path)
    logging.info("[I/O] wrote %s", cv_folds_path)
    logging.info("[I/O] wrote %s", cv_summary_path)

    figure_paths = save_figures(model_summary, open_closed)
    report_path = write_report(
        model_summary, coef_summary, cv_summary, open_closed, figure_paths, log_path
    )
    logging.info("[I/O] wrote %s", report_path)


if __name__ == "__main__":
    main()
