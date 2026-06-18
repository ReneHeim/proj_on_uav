#!/usr/bin/env python3
"""Test whether leaf inclination angle changes VZA-binned reflectance curves."""

from __future__ import annotations

import argparse
import logging
import math
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
import polars as pl
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
FEATURE_SOURCE = (
    ROOT
    / "outputs/result_01_reflectance_distributions/2024/ground_filtered/results/plot_week_angle_features_2024.parquet"
)
CONTRAST_SOURCE = (
    ROOT
    / "outputs/result_01_reflectance_distributions/2024/ground_filtered/results/matched_plot_contrasts_preliminary.parquet"
)
LIA_SOURCE = ROOT / "outputs/backup_metadata/csv/data/interim/2024/lia/20240912_lia_concatenated_average/Sheet1.csv"
LAI_SOURCE = ROOT / "outputs/backup_metadata/csv/data/processed/2024/quaratiello_giuseppe/LAI_DATA2/LAI_DATA_-_Copia.csv"
OUT_ROOT = ROOT / "outputs/backup_metadata/eda_lia_vza"
REPORTS_ROOT = ROOT / "outputs/reports"
LOG_ROOT = ROOT / "outputs/logs"

DATE_TO_WEEK = {
    "2024-06-17": 2,
    "2024-06-24": 3,
    "2024-07-08": 4,
    "2024-07-15": 5,
    "2024-07-25": 6,
    "2024-08-01": 7,
    "2024-08-06": 7,
    "2024-08-26": 8,
}

MAIN_BANDS = ["Red edge", "NIR"]
EARLY_STORY_WEEKS = [2, 3, 4, 5, 6, 7]
LATE_CAVEAT_WEEKS = [8]
PROJECT_TEAL = "#176B6B"
PROJECT_TEAL_LIGHT = "#76A9A9"
CULTIVAR_STYLES = {"aluco": ("#0072B2", "-"), "capone": ("#D55E00", "--")}
TREATMENT_STYLES = {"no_trt": ("#4D4D4D", "-"), "trt": ("#009E73", "--")}
LIA_GROUP_STYLES = {"low LIA": ("#4D4D4D", "--"), "high LIA": (PROJECT_TEAL, "-")}
LIA_TERCILE_COLORS = {"low LIA": "#4D4D4D", "mid LIA": PROJECT_TEAL_LIGHT, "high LIA": PROJECT_TEAL}
STRATIFICATIONS = {
    "cultivar": ["cult"],
    "treatment": ["trt"],
    "cultivar_treatment": ["cult", "trt"],
}
MODEL_FORMULAS = {
    "M0_controls": "reflectance ~ C(week) + C(cult) + C(trt)",
    "M1_vza": "reflectance ~ C(week) + C(cult) + C(trt) + C(vza_class)",
    "M2_vza_lia": "reflectance ~ C(week) + C(cult) + C(trt) + C(vza_class) + lia_z",
    "M3_vza_lia_interaction": "reflectance ~ C(week) + C(cult) + C(trt) + C(vza_class) * lia_z",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-source", type=Path, default=FEATURE_SOURCE)
    parser.add_argument("--contrast-source", type=Path, default=CONTRAST_SOURCE)
    parser.add_argument("--lia-source", type=Path, default=LIA_SOURCE)
    parser.add_argument("--lai-source", type=Path, default=LAI_SOURCE)
    parser.add_argument("--output-root", type=Path, default=OUT_ROOT)
    return parser.parse_args()


def configure_logging() -> Path:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    path = LOG_ROOT / f"analyze_lia_vza_binned_reflectance_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(path), logging.StreamHandler()],
        force=True,
    )
    return path


def phase(name: str, t0: float) -> None:
    logging.info("[PHASE] %s: %.1fs", name, time.perf_counter() - t0)


def style_axis(axis: plt.Axes, grid_axis: str = "y") -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis=grid_axis, color="#E3E6E8", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.tick_params(labelsize=8)


def set_local_ylim(axis: plt.Axes, values: pd.Series | np.ndarray, pad_fraction: float = 0.08) -> None:
    arr = pd.Series(values).dropna().to_numpy()
    if arr.size == 0:
        return
    ymin = float(np.min(arr))
    ymax = float(np.max(arr))
    span = ymax - ymin
    if span == 0:
        span = max(abs(ymax), 0.01)
    pad = span * pad_fraction
    axis.set_ylim(ymin - pad, ymax + pad)


def stratum_style(stratum_type: str, stratum: str) -> tuple[str, str]:
    if stratum_type == "cultivar":
        return CULTIVAR_STYLES.get(stratum, ("#4D4D4D", "-"))
    if stratum_type == "treatment":
        return TREATMENT_STYLES.get(stratum, ("#4D4D4D", "-"))
    if stratum_type == "cultivar_treatment":
        cultivar, _, treatment = stratum.partition(" | ")
        color, _ = CULTIVAR_STYLES.get(cultivar, ("#4D4D4D", "-"))
        _, linestyle = TREATMENT_STYLES.get(treatment, ("#4D4D4D", "-"))
        return color, linestyle
    return "#4D4D4D", "-"


def read_parquet(path: Path) -> pd.DataFrame:
    t0 = time.perf_counter()
    df = pl.read_parquet(path).to_pandas()
    logging.info("[I/O] read parquet %s rows=%d cols=%d time=%.2fs", path, df.shape[0], df.shape[1], time.perf_counter() - t0)
    return df


def load_lia_raw(path: Path) -> pd.DataFrame:
    t0 = time.perf_counter()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["week"] = df["date"].map(DATE_TO_WEEK)
    df["plot_id"] = (90024 - df["ifz_id"].astype(int)).map(lambda x: f"plot_{x}")
    df["lia"] = pd.to_numeric(df["lia"], errors="coerce")
    logging.info("[I/O] read LIA csv %s rows=%d cols=%d time=%.2fs", path, df.shape[0], df.shape[1], time.perf_counter() - t0)
    return df


def aggregate_lia(lia_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapped = lia_raw.dropna(subset=["week", "lia"]).copy()
    mapped["week"] = mapped["week"].astype(int)
    diagnostics = (
        mapped.groupby(["plot_id", "week", "cult", "trt"], as_index=False)
        .agg(
            lia_mean=("lia", "mean"),
            lia_sd=("lia", "std"),
            lia_min=("lia", "min"),
            lia_max=("lia", "max"),
            lia_median=("lia", "median"),
            lia_q25=("lia", lambda s: s.quantile(0.25)),
            lia_q75=("lia", lambda s: s.quantile(0.75)),
            n_lia_raw=("lia", "size"),
            n_lia_dates=("date", "nunique"),
            n_lia_timings=("timing", "nunique"),
            n_lia_reps=("rep", "nunique"),
        )
        .rename(columns={"lia_mean": "lia"})
    )
    diagnostics["lia_iqr"] = diagnostics["lia_q75"] - diagnostics["lia_q25"]
    diagnostics["lia_cv"] = diagnostics["lia_sd"] / diagnostics["lia"].abs()
    aggregate = diagnostics[["plot_id", "week", "cult", "trt", "lia", "lia_sd", "lia_iqr", "lia_cv", "n_lia_raw", "n_lia_dates"]].copy()
    return aggregate, diagnostics


def observed_high_low_effects(
    joined: pd.DataFrame,
    strata_cols: list[str] | None = None,
    stratum_type: str = "overall",
    min_plots: int = 6,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    strata_cols = strata_cols or []
    group_cols = strata_cols + ["band_name", "week", "vza_class", "vza_midpoint"]
    for keys, group in joined.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_cols, keys, strict=True))
        band = key_map["band_name"]
        week = key_map["week"]
        vza_class = key_map["vza_class"]
        vza_midpoint = key_map["vza_midpoint"]
        clean = group.dropna(subset=["reflectance", "lia"])
        if clean["plot_id"].nunique() < min_plots:
            continue
        low_cut = clean["lia"].quantile(0.33)
        high_cut = clean["lia"].quantile(0.67)
        low = clean[clean["lia"] <= low_cut]
        high = clean[clean["lia"] >= high_cut]
        if low.empty or high.empty:
            continue
        row = {
            "stratum_type": stratum_type,
            "stratum": "overall" if not strata_cols else " | ".join(str(key_map[col]) for col in strata_cols),
            "band_name": band,
            "week": int(week),
            "vza_class": vza_class,
            "vza_midpoint": float(vza_midpoint),
            "n_low_lia_plots": int(low["plot_id"].nunique()),
            "n_high_lia_plots": int(high["plot_id"].nunique()),
            "low_lia_median_reflectance": float(low["reflectance"].median()),
            "high_lia_median_reflectance": float(high["reflectance"].median()),
            "observed_high_minus_low_lia": float(high["reflectance"].median() - low["reflectance"].median()),
        }
        for col in strata_cols:
            row[col] = key_map[col]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["band_name", "week", "vza_midpoint"]) if rows else pd.DataFrame()


def stratified_observed_effects(joined: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for name, cols in STRATIFICATIONS.items():
        frames.append(observed_high_low_effects(joined, cols, name, min_plots=4))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_join(features: pd.DataFrame, lia: pd.DataFrame, lia_diagnostics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = features.copy()
    features["week"] = features["week"].astype(int)
    lia = lia.copy()
    lia["week"] = lia["week"].astype(int)

    joined = features.merge(lia, on=["plot_id", "week"], how="inner", suffixes=("", "_lia"), validate="many_to_one")
    joined["lia_z"] = (joined["lia"] - joined["lia"].mean()) / joined["lia"].std(ddof=0)
    joined["lia_tercile"] = pd.qcut(joined["lia"], q=3, labels=["low LIA", "mid LIA", "high LIA"], duplicates="drop")

    feature_keys = features[["plot_id", "week"]].drop_duplicates()
    lia_keys = lia[["plot_id", "week"]].drop_duplicates()
    coverage = feature_keys.merge(lia_keys.assign(has_lia=True), on=["plot_id", "week"], how="left")
    coverage["has_lia"] = coverage["has_lia"].eq(True)
    coverage = coverage.merge(
        features.groupby(["plot_id", "week"], as_index=False).agg(
            n_feature_rows=("reflectance", "size"),
            n_vza_bins=("vza_class", "nunique"),
            n_bands=("band_name", "nunique"),
        ),
        on=["plot_id", "week"],
        how="left",
    ).merge(
        lia_diagnostics[["plot_id", "week", "lia", "lia_sd", "lia_iqr", "lia_cv", "n_lia_raw", "n_lia_dates"]],
        on=["plot_id", "week"],
        how="left",
    )
    return joined, coverage


def fit_clustered(formula: str, frame: pd.DataFrame):
    t0 = time.perf_counter()
    model = smf.ols(formula, data=frame).fit(cov_type="cluster", cov_kwds={"groups": frame["plot_id"]})
    logging.info("[PHASE] fit formula=%s rows=%d: %.2fs", formula, frame.shape[0], time.perf_counter() - t0)
    return model


def model_ladder(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparisons: list[dict[str, object]] = []
    terms: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []

    for band, band_df in joined.groupby("band_name"):
        clean = band_df.dropna(subset=["reflectance", "lia_z", "vza_class", "week", "cult", "trt", "plot_id"]).copy()
        if clean["plot_id"].nunique() < 4:
            logging.warning("Skipping band=%s because cluster count is too low", band)
            continue
        fitted = {}
        for model_name, formula in MODEL_FORMULAS.items():
            try:
                model = fit_clustered(formula, clean)
                fitted[model_name] = model
                comparisons.append(
                    {
                        "band_name": band,
                        "model": model_name,
                        "formula": formula,
                        "n_rows": int(model.nobs),
                        "n_plots": int(clean["plot_id"].nunique()),
                        "r2": float(model.rsquared),
                        "adj_r2": float(model.rsquared_adj),
                        "aic": float(model.aic),
                        "bic": float(model.bic),
                    }
                )
            except Exception as exc:  # pragma: no cover - depends on data rank.
                logging.exception("Model failed band=%s model=%s", band, model_name)
                comparisons.append({"band_name": band, "model": model_name, "formula": formula, "error": str(exc)})

        m3 = fitted.get("M3_vza_lia_interaction")
        if m3 is None:
            continue
        for term in m3.params.index:
            if "lia_z" not in term:
                continue
            terms.append(
                {
                    "band_name": band,
                    "term": term,
                    "coef": float(m3.params[term]),
                    "std_err_cluster": float(m3.bse[term]),
                    "p_value_cluster": float(m3.pvalues[term]),
                    "conf_low": float(m3.conf_int().loc[term, 0]),
                    "conf_high": float(m3.conf_int().loc[term, 1]),
                }
            )

        low = clean["lia_z"].quantile(0.25)
        high = clean["lia_z"].quantile(0.75)
        pred_low = clean.copy()
        pred_high = clean.copy()
        pred_low["lia_z"] = low
        pred_high["lia_z"] = high
        pred = clean[["band_name", "week", "vza_class", "vza_midpoint"]].copy()
        pred["pred_low_lia_q25"] = m3.predict(pred_low)
        pred["pred_high_lia_q75"] = m3.predict(pred_high)
        pred["pred_high_minus_low_lia"] = pred["pred_high_lia_q75"] - pred["pred_low_lia_q25"]
        predictions.append(
            pred.groupby(["band_name", "week", "vza_class", "vza_midpoint"], as_index=False).mean(numeric_only=True)
        )

    pred_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    return pd.DataFrame(comparisons), pd.DataFrame(terms), pred_df


def plot_specific_slopes(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["plot_id", "week", "band_name"]
    for (plot_id, week, band), group in joined.groupby(group_cols):
        clean = group.dropna(subset=["reflectance", "vza_midpoint", "lia"])
        if clean["vza_midpoint"].nunique() < 3:
            continue
        slope, intercept = np.polyfit(clean["vza_midpoint"], clean["reflectance"], deg=1)
        rows.append(
            {
                "plot_id": plot_id,
                "week": int(week),
                "band_name": band,
                "cult": str(clean["cult"].iloc[0]),
                "trt": str(clean["trt"].iloc[0]),
                "cult_trt": f"{clean['cult'].iloc[0]} | {clean['trt'].iloc[0]}",
                "lia": float(clean["lia"].iloc[0]),
                "lia_sd": float(clean["lia_sd"].iloc[0]) if "lia_sd" in clean else math.nan,
                "n_vza_bins": int(clean["vza_midpoint"].nunique()),
                "angular_slope_per_degree": float(slope),
                "angular_intercept": float(intercept),
                "reflectance_range": float(clean["reflectance"].max() - clean["reflectance"].min()),
                "reflectance_mean": float(clean["reflectance"].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_plot_slopes(
    slopes: pd.DataFrame,
    strata_cols: list[str] | None = None,
    stratum_type: str = "overall",
    min_plots: int = 4,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    strata_cols = strata_cols or []
    group_cols = strata_cols + ["week", "band_name"]
    for keys, group in slopes.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_cols, keys, strict=True))
        clean = group.dropna(subset=["lia", "angular_slope_per_degree", "reflectance_range"])
        if clean["plot_id"].nunique() < min_plots:
            continue
        row = {
            "stratum_type": stratum_type,
            "stratum": "overall" if not strata_cols else " | ".join(str(key_map[col]) for col in strata_cols),
            "week": int(key_map["week"]),
            "band_name": key_map["band_name"],
            "n_plots": int(clean["plot_id"].nunique()),
            "corr_lia_angular_slope": float(clean["lia"].corr(clean["angular_slope_per_degree"])),
            "corr_lia_reflectance_range": float(clean["lia"].corr(clean["reflectance_range"])),
            "median_angular_slope": float(clean["angular_slope_per_degree"].median()),
            "median_reflectance_range": float(clean["reflectance_range"].median()),
        }
        for col in strata_cols:
            row[col] = key_map[col]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["week", "band_name"]) if rows else pd.DataFrame()


def stratified_slope_summaries(slopes: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for name, cols in STRATIFICATIONS.items():
        frames.append(summarize_plot_slopes(slopes, cols, name, min_plots=4))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def contrast_model(contrast_source: Path, lia: pd.DataFrame) -> pd.DataFrame:
    if not contrast_source.exists():
        return pd.DataFrame()
    contrasts = read_parquet(contrast_source)
    contrasts["week"] = contrasts["week"].astype(int)
    joined = contrasts.merge(lia[["plot_id", "week", "lia"]], on=["plot_id", "week"], how="inner", validate="many_to_one")
    joined["lia_z"] = (joined["lia"] - joined["lia"].mean()) / joined["lia"].std(ddof=0)
    rows: list[dict[str, object]] = []
    formula = "relative_contrast ~ C(week) + C(vza_class) * lia_z"
    for band, group in joined.groupby("band_name"):
        clean = group.dropna(subset=["relative_contrast", "lia_z", "vza_class", "plot_id"]).copy()
        if clean["plot_id"].nunique() < 4:
            continue
        try:
            model = fit_clustered(formula, clean)
        except Exception as exc:  # pragma: no cover - depends on data rank.
            rows.append({"band_name": band, "error": str(exc)})
            continue
        rows.append(
            {
                "band_name": band,
                "formula": formula,
                "n_rows": int(model.nobs),
                "n_plots": int(clean["plot_id"].nunique()),
                "r2": float(model.rsquared),
                "adj_r2": float(model.rsquared_adj),
                "aic": float(model.aic),
                "bic": float(model.bic),
                "min_lia_vza_interaction_p": float(
                    min([p for term, p in model.pvalues.items() if "lia_z" in term], default=np.nan)
                ),
            }
        )
    return pd.DataFrame(rows)


def lai_cvi_diagnostic(path: Path) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        return pd.DataFrame(), f"LAI source was not found: `{path}`."
    lai = pd.read_csv(path)
    lai["LAI"] = pd.to_numeric(lai["LAI"], errors="coerce")
    # A duplicated export stores LAI multiplied by 1000; this source is already in LAI units.
    lai["DATE"] = lai["DATE"].astype(str)
    summary = (
        lai.groupby("DATE", as_index=False)
        .agg(
            n_rows=("LAI", "size"),
            n_arr_plot=("ARR._PLOT", "nunique"),
            lai_mean=("LAI", "mean"),
            lai_sd=("LAI", "std"),
            lai_min=("LAI", "min"),
            lai_max=("LAI", "max"),
        )
        .sort_values("DATE")
    )
    summary["lai_cv"] = summary["lai_sd"] / summary["lai_mean"].abs()
    years = sorted({str(date)[:4] for date in lai["DATE"].dropna().unique()})
    note = (
        "The available LAI table has only 48 rows, 8 ARR._PLOT values, and dates "
        f"{', '.join(years)}. It is useful as a canopy-variability diagnostic, but it is not joined "
        "to the 2024 VZA reflectance models because it lacks the 2024 `ifz_id` plot key and matching dates."
    )
    return summary, note


def save_curve_figure(joined: pd.DataFrame, output: Path) -> None:
    plot_df = (
        joined[joined["band_name"].isin(MAIN_BANDS)]
        .groupby(["week", "band_name", "lia_tercile", "vza_midpoint"], as_index=False, observed=True)
        .agg(reflectance=("reflectance", "median"))
    )
    weeks = sorted(plot_df["week"].unique())
    bands = [band for band in MAIN_BANDS if band in set(plot_df["band_name"])]
    fig, axes = plt.subplots(len(bands), len(weeks), figsize=(2.8 * len(weeks), 3.3 * len(bands)), sharex=True, sharey=False)
    if len(bands) == 1:
        axes = np.array([axes])
    for row, band in enumerate(bands):
        for col, week in enumerate(weeks):
            ax = axes[row, col]
            subset = plot_df[(plot_df["band_name"] == band) & (plot_df["week"] == week)]
            for label, group in subset.groupby("lia_tercile", observed=True):
                ax.plot(
                    group["vza_midpoint"],
                    group["reflectance"],
                    marker="o",
                    markersize=3,
                    linewidth=1.45,
                    color=LIA_TERCILE_COLORS.get(str(label), "#4D4D4D"),
                    label=str(label),
                )
            ax.set_title(f"{band}, week {week}", fontsize=9)
            style_axis(ax)
            if col == 0:
                ax.set_ylabel("Median reflectance")
            if row == len(bands) - 1:
                ax.set_xlabel("VZA midpoint")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("VZA-binned reflectance curves by leaf inclination tercile", y=0.995, fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_heatmap(observed: pd.DataFrame, output: Path) -> None:
    plot_df = observed[observed["band_name"].isin(MAIN_BANDS)].copy()
    bands = [band for band in MAIN_BANDS if band in set(plot_df["band_name"])]
    fig, axes = plt.subplots(1, len(bands), figsize=(6 * len(bands), 4.5), squeeze=False)
    for idx, band in enumerate(bands):
        ax = axes[0, idx]
        subset = plot_df[plot_df["band_name"] == band]
        matrix = subset.pivot_table(index="week", columns="vza_class", values="observed_high_minus_low_lia", aggfunc="mean")
        vmax = float(np.nanmax(np.abs(matrix.values))) if matrix.size else 0.01
        image = ax.imshow(matrix.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{band}: observed high-low LIA")
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels(matrix.index)
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
        ax.set_xlabel("VZA bin")
        ax.set_ylabel("Week")
        style_axis(ax, grid_axis="both")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Reflectance difference")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_stratified_heatmaps(stratified: pd.DataFrame, stratum_type: str, output: Path) -> None:
    plot_df = stratified[(stratified["stratum_type"] == stratum_type) & stratified["band_name"].isin(MAIN_BANDS)].copy()
    if plot_df.empty:
        return
    strata = sorted(plot_df["stratum"].unique())
    bands = [band for band in MAIN_BANDS if band in set(plot_df["band_name"])]
    fig, axes = plt.subplots(
        len(strata),
        len(bands),
        figsize=(5.8 * len(bands), max(3.0, 2.45 * len(strata))),
        squeeze=False,
    )
    vmax = float(np.nanmax(np.abs(plot_df["observed_high_minus_low_lia"]))) if not plot_df.empty else 0.01
    image = None
    for row, stratum in enumerate(strata):
        for col, band in enumerate(bands):
            ax = axes[row, col]
            subset = plot_df[(plot_df["stratum"] == stratum) & (plot_df["band_name"] == band)]
            matrix = subset.pivot_table(index="week", columns="vza_class", values="observed_high_minus_low_lia", aggfunc="mean")
            if matrix.empty:
                ax.axis("off")
                continue
            image = ax.imshow(matrix.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_title(f"{stratum} | {band}", fontsize=10)
            ax.set_yticks(range(len(matrix.index)))
            ax.set_yticklabels(matrix.index)
            ax.set_xticks(range(len(matrix.columns)))
            ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8)
            ax.set_xlabel("VZA bin")
            ax.set_ylabel("Week")
            style_axis(ax, grid_axis="both")
    if image is not None:
        fig.subplots_adjust(right=0.88, top=0.93, hspace=0.65, wspace=0.22)
        cbar_ax = fig.add_axes([0.90, 0.16, 0.018, 0.70])
        fig.colorbar(image, cax=cbar_ax, label="High-LIA minus low-LIA reflectance")
    fig.suptitle(f"Observed LIA effect by {stratum_type.replace('_', ' ')}", y=0.995, fontsize=13)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_stratified_curve_summary(stratified: pd.DataFrame, stratum_type: str, output: Path) -> None:
    plot_df = stratified[(stratified["stratum_type"] == stratum_type) & stratified["band_name"].isin(MAIN_BANDS)].copy()
    if plot_df.empty:
        return
    summary = (
        plot_df.groupby(["stratum", "band_name", "vza_midpoint"], as_index=False)
        .agg(observed_high_minus_low_lia=("observed_high_minus_low_lia", "median"))
    )
    strata = sorted(summary["stratum"].unique())
    bands = [band for band in MAIN_BANDS if band in set(summary["band_name"])]
    fig, axes = plt.subplots(1, len(bands), figsize=(6 * len(bands), 4.2), squeeze=False)
    for col, band in enumerate(bands):
        ax = axes[0, col]
        subset = summary[summary["band_name"] == band]
        for stratum, group in subset.groupby("stratum"):
            color, linestyle = stratum_style(stratum_type, stratum)
            ax.plot(
                group["vza_midpoint"],
                group["observed_high_minus_low_lia"],
                marker="o",
                markersize=3,
                linewidth=1.5,
                color=color,
                linestyle=linestyle,
                label=stratum,
            )
        ax.axhline(0, color="#222222", linewidth=0.9)
        style_axis(ax)
        ax.set_title(f"{band}")
        ax.set_xlabel("VZA midpoint")
        ax.set_ylabel("Median high-LIA minus low-LIA reflectance")
    axes[0, -1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle(f"Median LIA effect curve by {stratum_type.replace('_', ' ')}", y=1.0, fontsize=13)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_story_strength_by_week(observed: pd.DataFrame, output: Path, weeks: list[int] | None = None) -> None:
    weeks = weeks or EARLY_STORY_WEEKS
    plot_df = observed[observed["band_name"].isin(MAIN_BANDS) & observed["week"].isin(weeks)].copy()
    if plot_df.empty:
        return
    summary = (
        plot_df.groupby(["band_name", "week"], as_index=False)
        .agg(
            median_effect=("observed_high_minus_low_lia", "median"),
            min_effect=("observed_high_minus_low_lia", "min"),
            max_effect=("observed_high_minus_low_lia", "max"),
        )
        .sort_values(["band_name", "week"])
    )
    bands = [band for band in MAIN_BANDS if band in set(summary["band_name"])]
    fig, axes = plt.subplots(1, len(bands), figsize=(5.8 * len(bands), 4.5), sharey=False, squeeze=False)
    for col, band in enumerate(bands):
        ax = axes[0, col]
        subset = summary[summary["band_name"] == band]
        ax.fill_between(
            subset["week"],
            subset["min_effect"],
            subset["max_effect"],
            color=PROJECT_TEAL_LIGHT,
            alpha=0.22,
            label="range across VZA bins",
        )
        ax.plot(
            subset["week"],
            subset["median_effect"],
            marker="o",
            markersize=4,
            linewidth=2.0,
            color=PROJECT_TEAL,
            label="median across VZA bins",
        )
        ax.axhline(0, color="#222222", linewidth=0.9)
        ax.set_title(band, fontsize=13)
        ax.set_xlabel("Week")
        ax.set_ylabel("High-LIA minus low-LIA reflectance")
        ax.set_xticks(sorted(summary["week"].unique()))
        style_axis(ax)
        strongest = subset.loc[subset["median_effect"].abs().idxmax()] if not subset.empty else None
        if strongest is not None:
            ax.annotate(
                f"largest early\nweek {int(strongest['week'])}",
                xy=(float(strongest["week"]), float(strongest["median_effect"])),
                xytext=(float(strongest["week"]) - 0.85, float(strongest["max_effect"]) * 0.65),
                arrowprops={"arrowstyle": "->", "linewidth": 1.1},
                fontsize=9,
            )
    axes[0, -1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle("Early-season leaf-angle effect, excluding late dead-canopy week 8", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_story_angular_profile(
    stratified: pd.DataFrame,
    stratum_type: str,
    output: Path,
    weeks: tuple[int, ...] = tuple(EARLY_STORY_WEEKS),
) -> None:
    plot_df = stratified[
        (stratified["stratum_type"] == stratum_type)
        & stratified["band_name"].isin(MAIN_BANDS)
        & stratified["week"].isin(weeks)
    ].copy()
    if plot_df.empty:
        return
    bands = [band for band in MAIN_BANDS if band in set(plot_df["band_name"])]
    fig, axes = plt.subplots(len(bands), len(weeks), figsize=(3.3 * len(weeks), 3.5 * len(bands)), sharex=True, squeeze=False)
    for row, band in enumerate(bands):
        for col, week in enumerate(weeks):
            ax = axes[row, col]
            subset = plot_df[(plot_df["band_name"] == band) & (plot_df["week"] == week)]
            for stratum, group in subset.groupby("stratum"):
                group = group.sort_values("vza_midpoint")
                color, linestyle = stratum_style(stratum_type, stratum)
                ax.plot(
                    group["vza_midpoint"],
                    group["observed_high_minus_low_lia"],
                    marker="o",
                    markersize=3,
                    linewidth=1.45,
                    color=color,
                    linestyle=linestyle,
                    label=stratum,
                )
            ax.axhline(0, color="#222222", linewidth=0.9)
            ax.set_title(f"{band}, week {week}", fontsize=12)
            ax.set_xlabel("VZA midpoint")
            style_axis(ax)
            if col == 0:
                ax.set_ylabel("High-LIA minus low-LIA reflectance")
            else:
                ax.set_ylabel("")
    axes[0, -1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, title=stratum_type.replace("_", " "))
    fig.suptitle(
        f"Progression of the leaf-angle angular effect across early weeks ({stratum_type.replace('_', ' ')})",
        fontsize=15,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_story_reflectance_curves(joined: pd.DataFrame, output: Path, week: int = 7) -> None:
    data = joined[(joined["week"] == week) & joined["band_name"].isin(MAIN_BANDS)].copy()
    if data.empty:
        return
    data["stratum"] = data["cult"].astype(str) + " | " + data["trt"].astype(str)
    labelled = []
    for (stratum, band), group in data.groupby(["stratum", "band_name"]):
        low_cut = group["lia"].quantile(0.33)
        high_cut = group["lia"].quantile(0.67)
        tmp = group[(group["lia"] <= low_cut) | (group["lia"] >= high_cut)].copy()
        tmp["lia_group"] = np.where(tmp["lia"] <= low_cut, "low LIA", "high LIA")
        labelled.append(tmp)
    if not labelled:
        return
    plot_df = pd.concat(labelled, ignore_index=True)
    summary = (
        plot_df.groupby(["stratum", "band_name", "lia_group", "vza_midpoint"], as_index=False)
        .agg(reflectance=("reflectance", "median"), n_plots=("plot_id", "nunique"))
    )
    strata = sorted(summary["stratum"].unique())
    bands = [band for band in MAIN_BANDS if band in set(summary["band_name"])]
    fig, axes = plt.subplots(len(strata), len(bands), figsize=(5.0 * len(bands), 2.7 * len(strata)), sharex=True, squeeze=False)
    for row, stratum in enumerate(strata):
        for col, band in enumerate(bands):
            ax = axes[row, col]
            subset = summary[(summary["stratum"] == stratum) & (summary["band_name"] == band)]
            for label, group in subset.groupby("lia_group"):
                group = group.sort_values("vza_midpoint")
                color, linestyle = LIA_GROUP_STYLES[label]
                ax.plot(
                    group["vza_midpoint"],
                    group["reflectance"],
                    marker="o",
                    markersize=3.2,
                    linewidth=1.65,
                    linestyle=linestyle,
                    label=label,
                    color=color,
                )
            n_low = subset.loc[subset["lia_group"] == "low LIA", "n_plots"].max()
            n_high = subset.loc[subset["lia_group"] == "high LIA", "n_plots"].max()
            ax.set_title(f"{stratum} | {band} (n={int(n_low)} low, {int(n_high)} high)", fontsize=10)
            ax.set_xlabel("VZA midpoint")
            ax.set_ylabel("Median reflectance")
            style_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(f"Raw early evidence in week {week}: high-LIA plots have different angular reflectance curves", fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def build_raw_lia_curve_summary(joined: pd.DataFrame, weeks: list[int] | None = None) -> pd.DataFrame:
    weeks = weeks or EARLY_STORY_WEEKS
    data = joined[joined["week"].isin(weeks) & joined["band_name"].isin(MAIN_BANDS)].copy()
    if data.empty:
        return pd.DataFrame()
    labelled = []
    group_cols = ["cult", "trt", "week", "band_name"]
    for _, group in data.groupby(group_cols):
        if group["plot_id"].nunique() < 4:
            continue
        low_cut = group["lia"].quantile(0.33)
        high_cut = group["lia"].quantile(0.67)
        tmp = group[(group["lia"] <= low_cut) | (group["lia"] >= high_cut)].copy()
        tmp["lia_group"] = np.where(tmp["lia"] <= low_cut, "low LIA", "high LIA")
        labelled.append(tmp)
    if not labelled:
        return pd.DataFrame()
    plot_df = pd.concat(labelled, ignore_index=True)
    return (
        plot_df.groupby(["cult", "trt", "week", "band_name", "lia_group", "vza_midpoint"], as_index=False)
        .agg(reflectance=("reflectance", "median"), n_plots=("plot_id", "nunique"))
        .sort_values(["cult", "band_name", "week", "trt", "vza_midpoint", "lia_group"])
    )


def save_raw_lia_curve_progression_by_cultivar_band(summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    if summary.empty:
        return []
    outputs = []
    weeks = [week for week in EARLY_STORY_WEEKS if week in set(summary["week"])]
    treatments = sorted(summary["trt"].unique())
    for cultivar in sorted(summary["cult"].unique()):
        for band in [band for band in MAIN_BANDS if band in set(summary["band_name"])]:
            subset = summary[(summary["cult"] == cultivar) & (summary["band_name"] == band)]
            if subset.empty:
                continue
            figure, axes = plt.subplots(
                len(weeks),
                len(treatments),
                figsize=(4.2 * len(treatments), 2.35 * len(weeks)),
                sharex=True,
                sharey=False,
                squeeze=False,
            )
            for row, week in enumerate(weeks):
                for col, treatment in enumerate(treatments):
                    axis = axes[row, col]
                    panel = subset[(subset["week"] == week) & (subset["trt"] == treatment)]
                    for lia_group in ["low LIA", "high LIA"]:
                        curve = panel[panel["lia_group"] == lia_group].sort_values("vza_midpoint")
                        if curve.empty:
                            continue
                        color, linestyle = LIA_GROUP_STYLES[lia_group]
                        axis.plot(
                            curve["vza_midpoint"],
                            curve["reflectance"],
                            marker="o",
                            markersize=2.8,
                            linewidth=1.45,
                            linestyle=linestyle,
                            color=color,
                            label=lia_group,
                        )
                    if row == 0:
                        axis.set_title(treatment, fontsize=10, fontweight="bold")
                    if col == 0:
                        axis.set_ylabel(f"Week {week}\nmedian reflectance")
                    if row == len(weeks) - 1:
                        axis.set_xlabel("VZA midpoint")
                    low_n = panel.loc[panel["lia_group"] == "low LIA", "n_plots"].max()
                    high_n = panel.loc[panel["lia_group"] == "high LIA", "n_plots"].max()
                    if not pd.isna(low_n) and not pd.isna(high_n):
                        axis.text(
                            0.98,
                            0.92,
                            f"n={int(low_n)} low, {int(high_n)} high",
                            ha="right",
                            va="top",
                            transform=axis.transAxes,
                            fontsize=7,
                            color="#555555",
                        )
                    style_axis(axis)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
            figure.suptitle(
                f"{cultivar.capitalize()} {band}: raw low/high LIA angular curves across early weeks",
                fontsize=13,
                y=1.035,
            )
            figure.tight_layout(rect=[0, 0, 1, 0.98])
            output = output_dir / f"03_raw_lia_curves_progression_{cultivar}_{band.lower().replace(' ', '_')}_2024.png"
            figure.savefig(output, dpi=240, bbox_inches="tight")
            plt.close(figure)
            outputs.append(output)
    return outputs


def build_plot_level_lia_curve_summary(joined: pd.DataFrame, weeks: list[int] | None = None) -> pd.DataFrame:
    weeks = weeks or EARLY_STORY_WEEKS
    data = joined[joined["week"].isin(weeks) & joined["band_name"].isin(MAIN_BANDS)].copy()
    if data.empty:
        return pd.DataFrame()
    labelled = []
    group_cols = ["cult", "trt", "week", "band_name"]
    for _, group in data.groupby(group_cols):
        if group["plot_id"].nunique() < 4:
            continue
        low_cut = group[["plot_id", "lia"]].drop_duplicates()["lia"].quantile(0.33)
        high_cut = group[["plot_id", "lia"]].drop_duplicates()["lia"].quantile(0.67)
        tmp = group.copy()
        tmp["lia_group"] = "mid LIA"
        tmp.loc[tmp["lia"] <= low_cut, "lia_group"] = "low LIA"
        tmp.loc[tmp["lia"] >= high_cut, "lia_group"] = "high LIA"
        labelled.append(tmp)
    if not labelled:
        return pd.DataFrame()
    plot_df = pd.concat(labelled, ignore_index=True)
    return (
        plot_df.groupby(["plot_id", "cult", "trt", "week", "band_name", "lia", "lia_group", "vza_midpoint"], as_index=False)
        .agg(reflectance=("reflectance", "median"))
        .sort_values(["cult", "band_name", "week", "trt", "plot_id", "vza_midpoint"])
    )


def save_plot_level_lia_curve_progression(summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    if summary.empty:
        return []
    outputs = []
    weeks = [week for week in EARLY_STORY_WEEKS if week in set(summary["week"])]
    treatments = sorted(summary["trt"].unique())
    styles = {
        "low LIA": ("#4D4D4D", "--", 0.70),
        "mid LIA": (PROJECT_TEAL_LIGHT, ":", 0.45),
        "high LIA": (PROJECT_TEAL, "-", 0.82),
    }
    for cultivar in sorted(summary["cult"].unique()):
        for band in [band for band in MAIN_BANDS if band in set(summary["band_name"])]:
            subset = summary[(summary["cult"] == cultivar) & (summary["band_name"] == band)]
            if subset.empty:
                continue
            figure, axes = plt.subplots(
                len(weeks),
                len(treatments),
                figsize=(4.2 * len(treatments), 2.35 * len(weeks)),
                sharex=True,
                sharey=False,
                squeeze=False,
            )
            for row, week in enumerate(weeks):
                for col, treatment in enumerate(treatments):
                    axis = axes[row, col]
                    panel = subset[(subset["week"] == week) & (subset["trt"] == treatment)]
                    for plot_id, curve in panel.groupby("plot_id"):
                        curve = curve.sort_values("vza_midpoint")
                        lia_group = str(curve["lia_group"].iloc[0])
                        color, linestyle, alpha = styles.get(lia_group, ("#777777", "-", 0.5))
                        axis.plot(
                            curve["vza_midpoint"],
                            curve["reflectance"],
                            linewidth=1.05,
                            linestyle=linestyle,
                            color=color,
                            alpha=alpha,
                            label=lia_group,
                        )
                    if row == 0:
                        axis.set_title(treatment, fontsize=10, fontweight="bold")
                    if col == 0:
                        axis.set_ylabel(f"Week {week}\nreflectance")
                    if row == len(weeks) - 1:
                        axis.set_xlabel("VZA midpoint")
                    counts = panel[["plot_id", "lia_group"]].drop_duplicates()["lia_group"].value_counts().to_dict()
                    axis.text(
                        0.98,
                        0.92,
                        f"plots: low {counts.get('low LIA', 0)}, mid {counts.get('mid LIA', 0)}, high {counts.get('high LIA', 0)}",
                        ha="right",
                        va="top",
                        transform=axis.transAxes,
                        fontsize=7,
                        color="#555555",
                    )
                    set_local_ylim(axis, panel["reflectance"])
                    style_axis(axis)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            unique = {}
            for handle, label in zip(handles, labels, strict=False):
                unique.setdefault(label, handle)
            ordered = [label for label in ["low LIA", "mid LIA", "high LIA"] if label in unique]
            figure.legend([unique[label] for label in ordered], ordered, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
            figure.suptitle(
                f"{cultivar.capitalize()} {band}: individual plot angular curves across early weeks",
                fontsize=13,
                y=1.035,
            )
            figure.tight_layout(rect=[0, 0, 1, 0.98])
            output = output_dir / f"03_plot_level_lia_curves_progression_{cultivar}_{band.lower().replace(' ', '_')}_2024.png"
            figure.savefig(output, dpi=240, bbox_inches="tight")
            plt.close(figure)
            outputs.append(output)
    return outputs


def save_plot_pair_lia_curve_progression(summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    if summary.empty:
        return []
    outputs = []
    weeks = [week for week in EARLY_STORY_WEEKS if week in set(summary["week"])]
    style_lookup = {
        "low LIA": ("#4D4D4D", "--"),
        "high LIA": (PROJECT_TEAL, "-"),
    }
    for cultivar in sorted(summary["cult"].unique()):
        for band in [band for band in MAIN_BANDS if band in set(summary["band_name"])]:
            for treatment in sorted(summary["trt"].unique()):
                subset = summary[
                    (summary["cult"] == cultivar)
                    & (summary["band_name"] == band)
                    & (summary["trt"] == treatment)
                    & (summary["lia_group"].isin(["low LIA", "high LIA"]))
                ].copy()
                if subset.empty:
                    continue

                pairs_by_week: dict[int, list[tuple[str, str]]] = {}
                max_pairs = 0
                for week in weeks:
                    panel = subset[subset["week"] == week]
                    low_plots = (
                        panel[panel["lia_group"] == "low LIA"][["plot_id", "lia"]]
                        .drop_duplicates()
                        .sort_values(["lia", "plot_id"])
                    )
                    high_plots = (
                        panel[panel["lia_group"] == "high LIA"][["plot_id", "lia"]]
                        .drop_duplicates()
                        .sort_values(["lia", "plot_id"])
                    )
                    pairs = list(zip(low_plots["plot_id"].to_list(), high_plots["plot_id"].to_list(), strict=False))
                    pairs_by_week[week] = pairs
                    max_pairs = max(max_pairs, len(pairs))
                if max_pairs == 0:
                    continue

                figure, axes = plt.subplots(
                    len(weeks),
                    max_pairs,
                    figsize=(4.3 * max_pairs, 2.35 * len(weeks)),
                    sharex=True,
                    sharey=False,
                    squeeze=False,
                )
                for row, week in enumerate(weeks):
                    panel = subset[subset["week"] == week]
                    for col in range(max_pairs):
                        axis = axes[row, col]
                        pairs = pairs_by_week.get(week, [])
                        if col >= len(pairs):
                            axis.axis("off")
                            continue
                        low_plot, high_plot = pairs[col]
                        pair_values = []
                        for lia_group, plot_id in [("low LIA", low_plot), ("high LIA", high_plot)]:
                            curve = panel[(panel["lia_group"] == lia_group) & (panel["plot_id"] == plot_id)].sort_values("vza_midpoint")
                            if curve.empty:
                                continue
                            color, linestyle = style_lookup[lia_group]
                            axis.plot(
                                curve["vza_midpoint"],
                                curve["reflectance"],
                                marker="o",
                                markersize=2.8,
                                linewidth=1.55,
                                linestyle=linestyle,
                                color=color,
                                label=lia_group,
                            )
                            pair_values.extend(curve["reflectance"].to_list())
                            lia_value = float(curve["lia"].iloc[0])
                            axis.text(
                                0.02,
                                0.92 if lia_group == "low LIA" else 0.82,
                                f"{lia_group}: {plot_id}, LIA={lia_value:.1f}",
                                ha="left",
                                va="top",
                                transform=axis.transAxes,
                                fontsize=6.8,
                                color=color,
                            )
                        if row == 0:
                            axis.set_title(f"pair {col + 1}", fontsize=10, fontweight="bold")
                        if col == 0:
                            axis.set_ylabel(f"Week {week}\nreflectance")
                        if row == len(weeks) - 1:
                            axis.set_xlabel("VZA midpoint")
                        set_local_ylim(axis, np.array(pair_values))
                        style_axis(axis)

                handles, labels = axes[0, 0].get_legend_handles_labels()
                figure.legend(
                    handles,
                    labels,
                    loc="upper center",
                    ncol=2,
                    frameon=False,
                    bbox_to_anchor=(0.5, 1.012),
                    fontsize=11,
                    handlelength=3.0,
                )
                figure.suptitle(
                    f"{cultivar.capitalize()} {band}, {treatment}: paired individual low/high LIA curves",
                    fontsize=13,
                    y=1.04,
                )
                figure.tight_layout(rect=[0, 0, 1, 0.98])
                output = (
                    output_dir
                    / f"03_pair_lia_curves_progression_{cultivar}_{treatment}_{band.lower().replace(' ', '_')}_2024.png"
                )
                figure.savefig(output, dpi=240, bbox_inches="tight")
                plt.close(figure)
                outputs.append(output)
    return outputs


def save_coeff_figure(terms: pd.DataFrame, output: Path) -> None:
    subset = terms[terms["band_name"].isin(MAIN_BANDS) & terms["term"].str.contains("C\\(vza_class\\).*:lia_z", regex=True)].copy()
    if subset.empty:
        return
    subset["label"] = subset["band_name"] + " | " + subset["term"].str.replace("C(vza_class)[T.", "", regex=False).str.replace("]:lia_z", "", regex=False)
    subset = subset.sort_values(["band_name", "coef"])
    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * subset.shape[0])))
    y = np.arange(subset.shape[0])
    ax.errorbar(
        subset["coef"],
        y,
        xerr=[subset["coef"] - subset["conf_low"], subset["conf_high"] - subset["coef"]],
        fmt="o",
        capsize=2,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(subset["label"], fontsize=8)
    ax.set_xlabel("Cluster-robust coefficient")
    ax.set_title("LIA x VZA interaction terms")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows._"
    view = df[columns].head(max_rows).copy()
    headers = list(view.columns)
    rows = []
    for record in view.to_dict(orient="records"):
        row = []
        for header in headers:
            value = record[header]
            if isinstance(value, float):
                row.append("" if math.isnan(value) else f"{value:.4f}")
            else:
                row.append(str(value))
        rows.append(row)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(
    output_root: Path,
    report_path: Path,
    joined: pd.DataFrame,
    coverage: pd.DataFrame,
    lia_diagnostics: pd.DataFrame,
    comparisons: pd.DataFrame,
    terms: pd.DataFrame,
    pred: pd.DataFrame,
    observed_effects: pd.DataFrame,
    stratified_effects: pd.DataFrame,
    slopes_summary: pd.DataFrame,
    stratified_slopes_summary: pd.DataFrame,
    contrast_summary: pd.DataFrame,
    lai_summary: pd.DataFrame,
    lai_note: str,
    log_path: Path,
) -> None:
    m2_m3 = comparisons.pivot_table(index="band_name", columns="model", values=["adj_r2", "aic", "bic"], aggfunc="first")
    delta_rows = []
    for band in sorted(comparisons["band_name"].dropna().unique()):
        row = {"band_name": band}
        try:
            row["delta_adj_r2_M3_minus_M2"] = float(m2_m3.loc[band, ("adj_r2", "M3_vza_lia_interaction")] - m2_m3.loc[band, ("adj_r2", "M2_vza_lia")])
            row["delta_aic_M3_minus_M2"] = float(m2_m3.loc[band, ("aic", "M3_vza_lia_interaction")] - m2_m3.loc[band, ("aic", "M2_vza_lia")])
            row["delta_bic_M3_minus_M2"] = float(m2_m3.loc[band, ("bic", "M3_vza_lia_interaction")] - m2_m3.loc[band, ("bic", "M2_vza_lia")])
        except Exception:
            continue
        delta_rows.append(row)
    deltas = pd.DataFrame(delta_rows)

    best_observed = observed_effects[observed_effects["band_name"].isin(MAIN_BANDS)].copy()
    if not best_observed.empty:
        best_observed = best_observed.loc[best_observed["observed_high_minus_low_lia"].abs().sort_values(ascending=False).index]

    missing = coverage[~coverage["has_lia"]]
    high_var = lia_diagnostics.sort_values("lia_iqr", ascending=False).head(10)
    stratified_top = stratified_effects[stratified_effects["band_name"].isin(MAIN_BANDS)].copy()
    if not stratified_top.empty:
        stratified_top = stratified_top.loc[stratified_top["observed_high_minus_low_lia"].abs().sort_values(ascending=False).index]

    lines = [
        "# LIA x VZA-Binned Reflectance",
        "",
        "## Results table",
        "",
        markdown_table(deltas, ["band_name", "delta_adj_r2_M3_minus_M2", "delta_aic_M3_minus_M2", "delta_bic_M3_minus_M2"]),
        "",
        "## Interpretation",
        "",
        "This analysis tests whether leaf inclination angle changes the reflectance curve across VZA bins. Evidence for directional scattering is strongest when the LIA x VZA interaction improves adjusted R2 or AIC/BIC and the observed high-LIA minus low-LIA effect changes across VZA bins and weeks.",
        "",
        "## Strongest observed high-LIA minus low-LIA effects",
        "",
        markdown_table(best_observed, ["band_name", "week", "vza_class", "vza_midpoint", "n_low_lia_plots", "n_high_lia_plots", "observed_high_minus_low_lia"], max_rows=20),
        "",
        "## Plot-specific behavior check",
        "",
        "Per-plot angular slopes were computed before any plot-level averaging across plots. These rows show whether LIA is associated with the slope or range of each plot's own VZA curve.",
        "",
        markdown_table(slopes_summary[slopes_summary["band_name"].isin(MAIN_BANDS)], ["week", "band_name", "n_plots", "corr_lia_angular_slope", "corr_lia_reflectance_range", "median_angular_slope", "median_reflectance_range"], max_rows=30),
        "",
        "## Stratified cultivar/treatment checks",
        "",
        "These tables are descriptive rather than formal subgroup inference because cultivar-treatment groups have only six plots each. They are included to expose plot-group-specific behavior instead of hiding it in the global average.",
        "",
        "Largest subgroup high-LIA minus low-LIA effects:",
        "",
        markdown_table(stratified_top, ["stratum_type", "stratum", "band_name", "week", "vza_class", "n_low_lia_plots", "n_high_lia_plots", "observed_high_minus_low_lia"], max_rows=24),
        "",
        "Subgroup plot-slope correlations:",
        "",
        markdown_table(stratified_slopes_summary[stratified_slopes_summary["band_name"].isin(MAIN_BANDS)], ["stratum_type", "stratum", "week", "band_name", "n_plots", "corr_lia_angular_slope", "corr_lia_reflectance_range"], max_rows=24),
        "",
        "## Join and aggregation diagnostics",
        "",
        f"- Joined reflectance rows: `{joined.shape[0]}`",
        f"- Joined plot-week pairs: `{joined[['plot_id', 'week']].drop_duplicates().shape[0]}`",
        f"- Reflectance plot-week pairs without mapped LIA: `{missing.shape[0]}`",
        f"- Raw LIA rows used after week mapping: `{int(lia_diagnostics['n_lia_raw'].sum())}`",
        "",
        "Highest within plot-week LIA IQR values:",
        "",
        markdown_table(high_var, ["plot_id", "week", "cult", "trt", "lia", "lia_sd", "lia_iqr", "lia_cv", "n_lia_raw", "n_lia_dates"], max_rows=10),
        "",
        "## LAI CVI diagnostic",
        "",
        lai_note,
        "",
        markdown_table(lai_summary, ["DATE", "n_rows", "n_arr_plot", "lai_mean", "lai_sd", "lai_cv", "lai_min", "lai_max"], max_rows=20),
        "",
        "## Secondary contrast model",
        "",
        markdown_table(contrast_summary, ["band_name", "n_rows", "n_plots", "adj_r2", "aic", "bic", "min_lia_vza_interaction_p"], max_rows=10),
        "",
        "## Outputs",
        "",
        f"- Joined table: `{output_root / 'joined/lia_vza_reflectance_join_2024.csv'}`",
        f"- Join coverage: `{output_root / 'diagnostics/lia_vza_join_coverage_2024.csv'}`",
        f"- Raw LIA aggregation diagnostics: `{output_root / 'diagnostics/lia_plot_week_raw_variability_2024.csv'}`",
        f"- Model comparison: `{output_root / 'results/lia_vza_model_comparison_2024.csv'}`",
        f"- Interaction terms: `{output_root / 'results/lia_vza_interaction_terms_2024.csv'}`",
        f"- Predicted high-low LIA effect: `{output_root / 'results/lia_vza_predicted_high_low_effect_2024.csv'}`",
        f"- Observed high-low LIA effect: `{output_root / 'results/lia_vza_observed_high_low_effect_2024.csv'}`",
        f"- Stratified observed high-low LIA effect: `{output_root / 'results/lia_vza_stratified_observed_high_low_effect_2024.csv'}`",
        f"- Raw LIA curve progression summary: `{output_root / 'results/lia_vza_raw_lia_curve_progression_summary_2024.csv'}`",
        f"- Plot-level LIA curve progression summary: `{output_root / 'results/lia_vza_plot_level_lia_curve_progression_summary_2024.csv'}`",
        f"- Plot-specific slopes: `{output_root / 'results/lia_vza_plot_specific_slopes_2024.csv'}`",
        f"- Plot-specific slope summary: `{output_root / 'results/lia_vza_plot_specific_slope_summary_2024.csv'}`",
        f"- Stratified plot-specific slope summary: `{output_root / 'results/lia_vza_stratified_plot_specific_slope_summary_2024.csv'}`",
        f"- LAI CVI diagnostic: `{output_root / 'diagnostics/lai_cvi_diagnostic.csv'}`",
        f"- Curve figure: `{output_root / 'figures/lia_tercile_vza_curves_nir_rededge_2024.png'}`",
        f"- Heatmap figure: `{output_root / 'figures/lia_high_low_effect_heatmap_nir_rededge_2024.png'}`",
        f"- Cultivar heatmap: `{output_root / 'figures/lia_effect_heatmap_by_cultivar_nir_rededge_2024.png'}`",
        f"- Treatment heatmap: `{output_root / 'figures/lia_effect_heatmap_by_treatment_nir_rededge_2024.png'}`",
        f"- Cultivar-treatment heatmap: `{output_root / 'figures/lia_effect_heatmap_by_cultivar_treatment_nir_rededge_2024.png'}`",
        f"- Cultivar curves: `{output_root / 'figures/lia_effect_curve_by_cultivar_nir_rededge_2024.png'}`",
        f"- Treatment curves: `{output_root / 'figures/lia_effect_curve_by_treatment_nir_rededge_2024.png'}`",
        f"- Cultivar-treatment curves: `{output_root / 'figures/lia_effect_curve_by_cultivar_treatment_nir_rededge_2024.png'}`",
        f"- Story figure 1, early-season effect strength: `{output_root / 'figures/story/01_early_when_lia_matters_effect_strength_by_week_2024.png'}`",
        f"- Story figure 2, early angular profile by cultivar: `{output_root / 'figures/story/02_early_how_lia_effect_changes_with_vza_by_cultivar_2024.png'}`",
        f"- Story figure 2, early angular profile by treatment: `{output_root / 'figures/story/02_early_how_lia_effect_changes_with_vza_by_treatment_2024.png'}`",
        f"- Story figure 2, early angular profile by cultivar-treatment: `{output_root / 'figures/story/02_early_how_lia_effect_changes_with_vza_by_cultivar_treatment_2024.png'}`",
        f"- Story figure 3, raw week 7 reflectance curves: `{output_root / 'figures/story/03_raw_week7_reflectance_curves_by_cultivar_treatment_2024.png'}`",
        f"- Story figure 3, Aluco Red edge all-week raw curves: `{output_root / 'figures/story/03_raw_lia_curves_progression_aluco_red_edge_2024.png'}`",
        f"- Story figure 3, Aluco NIR all-week raw curves: `{output_root / 'figures/story/03_raw_lia_curves_progression_aluco_nir_2024.png'}`",
        f"- Story figure 3, Capone Red edge all-week raw curves: `{output_root / 'figures/story/03_raw_lia_curves_progression_capone_red_edge_2024.png'}`",
        f"- Story figure 3, Capone NIR all-week raw curves: `{output_root / 'figures/story/03_raw_lia_curves_progression_capone_nir_2024.png'}`",
        f"- Recommended plot-level figure, Aluco Red edge: `{output_root / 'figures/story/03_plot_level_lia_curves_progression_aluco_red_edge_2024.png'}`",
        f"- Recommended plot-level figure, Aluco NIR: `{output_root / 'figures/story/03_plot_level_lia_curves_progression_aluco_nir_2024.png'}`",
        f"- Recommended plot-level figure, Capone Red edge: `{output_root / 'figures/story/03_plot_level_lia_curves_progression_capone_red_edge_2024.png'}`",
        f"- Recommended plot-level figure, Capone NIR: `{output_root / 'figures/story/03_plot_level_lia_curves_progression_capone_nir_2024.png'}`",
        f"- Best readability, paired Aluco no_trt NIR: `{output_root / 'figures/story/03_pair_lia_curves_progression_aluco_no_trt_nir_2024.png'}`",
        f"- Best readability, paired Aluco trt NIR: `{output_root / 'figures/story/03_pair_lia_curves_progression_aluco_trt_nir_2024.png'}`",
        f"- Best readability, paired Capone no_trt NIR: `{output_root / 'figures/story/03_pair_lia_curves_progression_capone_no_trt_nir_2024.png'}`",
        f"- Best readability, paired Capone trt NIR: `{output_root / 'figures/story/03_pair_lia_curves_progression_capone_trt_nir_2024.png'}`",
        f"- Story figure 4, week 8 late/dead-canopy caveat by cultivar: `{output_root / 'figures/story/04_late_week8_dead_canopy_lia_effect_by_cultivar_2024.png'}`",
        f"- Story figure 4, week 8 late/dead-canopy caveat by treatment: `{output_root / 'figures/story/04_late_week8_dead_canopy_lia_effect_by_treatment_2024.png'}`",
        f"- Story figure 4, week 8 late/dead-canopy caveat by cultivar-treatment: `{output_root / 'figures/story/04_late_week8_dead_canopy_lia_effect_by_cultivar_treatment_2024.png'}`",
        f"- Coefficient figure: `{output_root / 'figures/lia_vza_interaction_coefficients_2024.png'}`",
        f"- Log: `{log_path}`",
        "",
        "## Reproducibility",
        "",
        "- Year: 2024",
        "- Reflectance source: ground-filtered VZA-binned plot-week table",
        "- Model: OLS with cluster-robust standard errors by `plot_id`",
        "- Main formula: `reflectance ~ C(week) + C(cult) + C(trt) + C(vza_class) * lia_z`",
        "- LIA date mapping: `2024-06-17->2`, `2024-06-24->3`, `2024-07-08->4`, `2024-07-15->5`, `2024-07-25->6`, `2024-08-01->7`, `2024-08-06->7`, `2024-08-26->8`",
        "- Random seed: not used",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    log_path = configure_logging()
    total = time.perf_counter()
    for subdir in ["joined", "results", "diagnostics", "figures", "reports"]:
        (args.output_root / subdir).mkdir(parents=True, exist_ok=True)
    (args.output_root / "figures/story").mkdir(parents=True, exist_ok=True)
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    features = read_parquet(args.feature_source)
    lia_raw = load_lia_raw(args.lia_source)
    phase("data loading", t0)

    t0 = time.perf_counter()
    lia, lia_diagnostics = aggregate_lia(lia_raw)
    joined, coverage = build_join(features, lia, lia_diagnostics)
    phase("join and preprocessing", t0)

    t0 = time.perf_counter()
    comparisons, terms, pred = model_ladder(joined)
    contrast_summary = contrast_model(args.contrast_source, lia)
    phase("model fitting", t0)

    t0 = time.perf_counter()
    slopes = plot_specific_slopes(joined)
    slopes_summary = summarize_plot_slopes(slopes)
    observed_effects = observed_high_low_effects(joined)
    stratified_effects = stratified_observed_effects(joined)
    stratified_slopes_summary = stratified_slope_summaries(slopes)
    raw_lia_curve_summary = build_raw_lia_curve_summary(joined)
    plot_level_lia_curve_summary = build_plot_level_lia_curve_summary(joined)
    lai_summary, lai_note = lai_cvi_diagnostic(args.lai_source)
    phase("plot-specific and LAI diagnostics", t0)

    t0 = time.perf_counter()
    joined.to_csv(args.output_root / "joined/lia_vza_reflectance_join_2024.csv", index=False)
    coverage.to_csv(args.output_root / "diagnostics/lia_vza_join_coverage_2024.csv", index=False)
    lia_diagnostics.to_csv(args.output_root / "diagnostics/lia_plot_week_raw_variability_2024.csv", index=False)
    comparisons.to_csv(args.output_root / "results/lia_vza_model_comparison_2024.csv", index=False)
    terms.to_csv(args.output_root / "results/lia_vza_interaction_terms_2024.csv", index=False)
    pred.to_csv(args.output_root / "results/lia_vza_predicted_high_low_effect_2024.csv", index=False)
    observed_effects.to_csv(args.output_root / "results/lia_vza_observed_high_low_effect_2024.csv", index=False)
    stratified_effects.to_csv(args.output_root / "results/lia_vza_stratified_observed_high_low_effect_2024.csv", index=False)
    raw_lia_curve_summary.to_csv(args.output_root / "results/lia_vza_raw_lia_curve_progression_summary_2024.csv", index=False)
    plot_level_lia_curve_summary.to_csv(args.output_root / "results/lia_vza_plot_level_lia_curve_progression_summary_2024.csv", index=False)
    slopes.to_csv(args.output_root / "results/lia_vza_plot_specific_slopes_2024.csv", index=False)
    slopes_summary.to_csv(args.output_root / "results/lia_vza_plot_specific_slope_summary_2024.csv", index=False)
    stratified_slopes_summary.to_csv(args.output_root / "results/lia_vza_stratified_plot_specific_slope_summary_2024.csv", index=False)
    contrast_summary.to_csv(args.output_root / "results/lia_vza_contrast_model_2024.csv", index=False)
    lai_summary.to_csv(args.output_root / "diagnostics/lai_cvi_diagnostic.csv", index=False)
    phase("write tables", t0)

    t0 = time.perf_counter()
    save_curve_figure(joined, args.output_root / "figures/lia_tercile_vza_curves_nir_rededge_2024.png")
    save_heatmap(observed_effects, args.output_root / "figures/lia_high_low_effect_heatmap_nir_rededge_2024.png")
    save_coeff_figure(terms, args.output_root / "figures/lia_vza_interaction_coefficients_2024.png")
    for stratum_type in STRATIFICATIONS:
        save_stratified_heatmaps(
            stratified_effects,
            stratum_type,
            args.output_root / f"figures/lia_effect_heatmap_by_{stratum_type}_nir_rededge_2024.png",
        )
        save_stratified_curve_summary(
            stratified_effects,
            stratum_type,
            args.output_root / f"figures/lia_effect_curve_by_{stratum_type}_nir_rededge_2024.png",
        )
    save_story_strength_by_week(
        observed_effects,
        args.output_root / "figures/story/01_early_when_lia_matters_effect_strength_by_week_2024.png",
    )
    for stratum_type in STRATIFICATIONS:
        save_story_angular_profile(
            stratified_effects,
            stratum_type,
            args.output_root / f"figures/story/02_early_how_lia_effect_changes_with_vza_by_{stratum_type}_2024.png",
        )
        save_story_angular_profile(
            stratified_effects,
            stratum_type,
            args.output_root / f"figures/story/04_late_week8_dead_canopy_lia_effect_by_{stratum_type}_2024.png",
            weeks=tuple(LATE_CAVEAT_WEEKS),
        )
    save_story_reflectance_curves(
        joined,
        args.output_root / "figures/story/03_raw_week7_reflectance_curves_by_cultivar_treatment_2024.png",
        week=7,
    )
    save_raw_lia_curve_progression_by_cultivar_band(
        raw_lia_curve_summary,
        args.output_root / "figures/story",
    )
    save_plot_level_lia_curve_progression(
        plot_level_lia_curve_summary,
        args.output_root / "figures/story",
    )
    save_plot_pair_lia_curve_progression(
        plot_level_lia_curve_summary,
        args.output_root / "figures/story",
    )
    phase("figures", t0)

    t0 = time.perf_counter()
    report_path = args.output_root / "reports/lia_vza_reflectance_summary.md"
    canonical_report = REPORTS_ROOT / "analyze_lia_vza_binned_reflectance_summary.md"
    write_report(
        args.output_root,
        report_path,
        joined,
        coverage,
        lia_diagnostics,
        comparisons,
        terms,
        pred,
        observed_effects,
        stratified_effects,
        slopes_summary,
        stratified_slopes_summary,
        contrast_summary,
        lai_summary,
        lai_note,
        log_path,
    )
    canonical_report.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    phase("report", t0)

    logging.info("Joined rows=%d plot-week pairs=%d", joined.shape[0], joined[["plot_id", "week"]].drop_duplicates().shape[0])
    logging.info("[PHASE] total: %.1fs", time.perf_counter() - total)


if __name__ == "__main__":
    main()
