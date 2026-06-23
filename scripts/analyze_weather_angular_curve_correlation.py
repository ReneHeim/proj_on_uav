"""Correlate 2024 weather with VZA, RAA, and phase-angle curve metrics."""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "outputs/backup_metadata/weather_angular_curve_correlation"
RESULT_DIR = OUT_ROOT / "results"
REPORT_DIR = OUT_ROOT / "reports"
LOG_DIR = ROOT / "outputs/logs"

WEATHER_PATH = Path(
    "/run/media/davidem/heim_data/Backup/proj_on_cerco/code/scripts/r/2024_leaf_dynamics/data/20250129_weather_indices.csv"
)
VZA_METRICS_PATH = (
    ROOT
    / "outputs/result_03_vza_curve_shape_metrics/vza_curve_shape_metrics_by_year_week_cultivar_band.csv"
)
RAA_PERCENT_PATH = (
    ROOT
    / "outputs/result_01_raa_sun_geometry/2024/ground_filtered/results/raa_vza_nadir_percent_change_summary_2024_trimmed.csv"
)
PHASE_PERCENT_PATH = (
    ROOT
    / "outputs/result_01_raa_sun_geometry/2024/ground_filtered/results/raa_phase_nadir_percent_change_summary_2024_trimmed.csv"
)

WEEK_DATES_2024 = {
    0: "2024-06-03",
    2: "2024-06-22",
    3: "2024-06-24",
    4: "2024-07-08",
    5: "2024-07-15",
    6: "2024-07-23",
    7: "2024-07-30",
    8: "2024-08-26",
}

WEATHER_VARIABLES = [
    "day_t_mean",
    "day_rh_mean",
    "day_globrad_mean",
    "day_prec_sum",
    "day_vpd_mean",
    "day_et0_sum",
    "w3_t_mean",
    "w3_globrad_mean",
    "w3_prec_sum",
    "w3_vpd_mean",
    "w7_t_mean",
    "w7_globrad_mean",
    "w7_prec_sum",
    "w7_vpd_mean",
]

CURVE_METRICS = [
    "vza_curve_amplitude",
    "vza_abs_high_minus_low",
    "vza_abs_slope",
    "vza_abs_curvature",
    "vza_quadratic_r2",
    "raa_percent_abs_median",
    "raa_percent_max_abs",
    "phase_percent_range",
    "phase_abs_slope",
    "phase_low_minus_high",
]


def setup_logging() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"analyze_weather_angular_curve_correlation_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    return log_path


def phase_slope(frame: pd.DataFrame) -> float:
    data = frame[["phase_midpoint", "median_percent_change"]].dropna()
    if len(data) < 3 or data["phase_midpoint"].nunique() < 3:
        return math.nan
    x = data["phase_midpoint"].to_numpy(float)
    y = data["median_percent_change"].to_numpy(float)
    return float(np.polyfit(x, y, 1)[0])


def pearson_r(x: pd.Series, y: pd.Series) -> float:
    data = pd.concat([x, y], axis=1).dropna()
    if len(data) < 3:
        return math.nan
    if data.iloc[:, 0].nunique() < 2 or data.iloc[:, 1].nunique() < 2:
        return math.nan
    return float(data.iloc[:, 0].corr(data.iloc[:, 1], method="pearson"))


def spearman_r(x: pd.Series, y: pd.Series) -> float:
    data = pd.concat([x, y], axis=1).dropna()
    if len(data) < 3:
        return math.nan
    if data.iloc[:, 0].nunique() < 2 or data.iloc[:, 1].nunique() < 2:
        return math.nan
    return float(data.iloc[:, 0].corr(data.iloc[:, 1], method="spearman"))


def build_weather_table() -> pd.DataFrame:
    t0 = time.time()
    weather = pd.read_csv(WEATHER_PATH)
    weather["datetime"] = pd.to_datetime(weather["datetime"], errors="coerce")
    weather = weather.dropna(subset=["datetime"]).copy()
    weather["date"] = weather["datetime"].dt.date

    daily = (
        weather.groupby("date", as_index=False)
        .agg(
            day_t_mean=("t_mean", "mean"),
            day_rh_mean=("rh_mean", "mean"),
            day_globrad_mean=("globrad_mean", "mean"),
            day_globrad_sum=("globrad_mean", "sum"),
            day_prec_sum=("prec_mean", "sum"),
            day_vpd_mean=("vpd", "mean"),
            day_et0_sum=("et0_hargreaves", "sum"),
        )
        .sort_values("date")
    )
    daily["date"] = pd.to_datetime(daily["date"])

    rows = []
    for week, date_string in WEEK_DATES_2024.items():
        date = pd.Timestamp(date_string)
        row = {"year": 2024, "week": week, "flight_date": date.date().isoformat()}
        same_day = daily[daily["date"] == date]
        if not same_day.empty:
            row.update(same_day.iloc[0].drop(labels=["date"]).to_dict())
        for days in (3, 7):
            window = daily[
                (daily["date"] >= date - pd.Timedelta(days=days - 1)) & (daily["date"] <= date)
            ]
            row[f"w{days}_t_mean"] = window["day_t_mean"].mean()
            row[f"w{days}_rh_mean"] = window["day_rh_mean"].mean()
            row[f"w{days}_globrad_mean"] = window["day_globrad_mean"].mean()
            row[f"w{days}_prec_sum"] = window["day_prec_sum"].sum()
            row[f"w{days}_vpd_mean"] = window["day_vpd_mean"].mean()
            row[f"w{days}_et0_sum"] = window["day_et0_sum"].sum()
        rows.append(row)

    result = pd.DataFrame(rows)
    logging.info(f"[PHASE] build_weather_table: {time.time() - t0:.1f}s")
    return result


def build_curve_metrics() -> pd.DataFrame:
    t0 = time.time()
    vza = pd.read_csv(VZA_METRICS_PATH)
    vza = vza[vza["year"] == 2024].copy()
    vza_metrics = vza.groupby(["year", "week", "band", "band_name"], as_index=False).agg(
        vza_curve_amplitude=("curve_amplitude", "mean"),
        vza_abs_high_minus_low=("high_minus_low_reflectance", lambda s: s.abs().mean()),
        vza_abs_slope=("linear_slope_per_degree", lambda s: s.abs().mean()),
        vza_abs_curvature=("quadratic_curvature", lambda s: s.abs().mean()),
        vza_quadratic_r2=("quadratic_r2", "mean"),
    )

    raa = pd.read_csv(RAA_PERCENT_PATH)
    raa_metrics = raa.groupby(["year", "week", "band", "band_name"], as_index=False).agg(
        raa_percent_abs_median=("median_percent_change", lambda s: s.abs().median()),
        raa_percent_max_abs=("median_percent_change", lambda s: s.abs().max()),
    )

    phase = pd.read_csv(PHASE_PERCENT_PATH)
    phase_by_bin = phase.groupby(
        ["year", "week", "band", "band_name", "phase_class"], as_index=False
    ).agg(
        phase_midpoint=("phase_midpoint", "mean"),
        median_percent_change=("median_percent_change", "median"),
        plots=("plots", "sum"),
    )
    phase_metrics = (
        phase_by_bin.groupby(["year", "week", "band", "band_name"], as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "phase_percent_range": g["median_percent_change"].max()
                    - g["median_percent_change"].min(),
                    "phase_abs_slope": abs(phase_slope(g)),
                    "phase_low_minus_high": (
                        g.loc[g["phase_midpoint"] <= 20, "median_percent_change"].mean()
                        - g.loc[g["phase_midpoint"] >= 50, "median_percent_change"].mean()
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )

    curves = vza_metrics.merge(raa_metrics, on=["year", "week", "band", "band_name"], how="outer")
    curves = curves.merge(phase_metrics, on=["year", "week", "band", "band_name"], how="outer")
    logging.info(f"[PHASE] build_curve_metrics: {time.time() - t0:.1f}s")
    return curves


def build_correlations(joined: pd.DataFrame) -> pd.DataFrame:
    t0 = time.time()
    rows = []
    for band_name, band_df in joined.groupby("band_name"):
        for weather_var in WEATHER_VARIABLES:
            for curve_metric in CURVE_METRICS:
                rows.append(
                    {
                        "band_name": band_name,
                        "weather_variable": weather_var,
                        "curve_metric": curve_metric,
                        "n_weeks": int(band_df[[weather_var, curve_metric]].dropna().shape[0]),
                        "pearson_r": pearson_r(band_df[weather_var], band_df[curve_metric]),
                        "spearman_r": spearman_r(band_df[weather_var], band_df[curve_metric]),
                    }
                )
    result = pd.DataFrame(rows)
    result["abs_pearson_r"] = result["pearson_r"].abs()
    result["abs_spearman_r"] = result["spearman_r"].abs()
    result = result.sort_values(["abs_spearman_r", "abs_pearson_r"], ascending=False)
    logging.info(f"[PHASE] build_correlations: {time.time() - t0:.1f}s")
    return result


def write_report(
    correlations: pd.DataFrame, joined: pd.DataFrame, outputs: dict[str, Path], log_path: Path
) -> None:
    top = correlations[correlations["n_weeks"] >= 6].head(15).copy()
    selected = correlations[
        (correlations["n_weeks"] >= 6)
        & (correlations["band_name"].isin(["Red edge", "NIR"]))
        & (
            correlations["curve_metric"].isin(
                [
                    "vza_curve_amplitude",
                    "raa_percent_abs_median",
                    "phase_percent_range",
                    "phase_low_minus_high",
                ]
            )
        )
    ].sort_values(["band_name", "curve_metric", "abs_spearman_r"], ascending=[True, True, False])

    def markdown_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "_No rows._"
        frame = frame.copy()
        columns = list(frame.columns)
        lines = [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        for _, row in frame.iterrows():
            values = []
            for col in columns:
                value = row[col]
                if isinstance(value, float):
                    values.append("" if pd.isna(value) else f"{value:.3f}")
                else:
                    values.append("" if pd.isna(value) else str(value))
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    lines = [
        "## Results: Weather vs Angular Curve Correlation",
        "",
        "Exploratory 2024 week-level correlations between flight-date weather and angular reflectance curve metrics.",
        "The sample size is small because each week has one weather context, so these values should be treated as hypothesis-generating, not causal proof.",
        "",
        "### Strongest correlations, all bands/metrics",
        "",
        markdown_table(
            top[
                [
                    "band_name",
                    "weather_variable",
                    "curve_metric",
                    "n_weeks",
                    "pearson_r",
                    "spearman_r",
                ]
            ]
        ),
        "",
        "### Selected paper-relevant Red Edge/NIR metrics",
        "",
        markdown_table(
            selected[
                [
                    "band_name",
                    "weather_variable",
                    "curve_metric",
                    "n_weeks",
                    "pearson_r",
                    "spearman_r",
                ]
            ].head(30)
        ),
        "",
        "**Interpretation**: Weather is correlated with some angular-curve descriptors, especially phase-angle and RAA contrast metrics, but the week-level sample size is limited. The strongest practical use is as a covariate/sensitivity check: report whether VZA/RAA/phase effects remain after accounting for flight-date temperature, radiation, precipitation, and VPD.",
        "",
        "### Reproducibility",
        "",
        f"- Flight weeks: `{sorted(joined['week'].dropna().unique().tolist())}`",
        f"- Weather source: `{WEATHER_PATH}`",
        f"- VZA metric source: `{VZA_METRICS_PATH}`",
        f"- RAA source: `{RAA_PERCENT_PATH}`",
        f"- Phase source: `{PHASE_PERCENT_PATH}`",
        f"- Log: `{log_path}`",
        "",
        "### Outputs",
        "",
    ]
    for label, path in outputs.items():
        lines.append(f"- {label}: `{path}`")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    outputs["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging()
    start = time.time()

    weather = build_weather_table()
    curves = build_curve_metrics()
    joined = curves.merge(weather, on=["year", "week"], how="left")
    correlations = build_correlations(joined)

    outputs = {
        "weather_week_table": RESULT_DIR / "weather_by_flight_week_2024.csv",
        "curve_metric_table": RESULT_DIR / "angular_curve_metrics_by_week_band_2024.csv",
        "joined_table": RESULT_DIR / "weather_angular_curve_join_2024.csv",
        "correlation_table": RESULT_DIR / "weather_angular_curve_correlations_2024.csv",
        "report": REPORT_DIR / "weather_angular_curve_correlation_summary.md",
    }
    weather.to_csv(outputs["weather_week_table"], index=False)
    curves.to_csv(outputs["curve_metric_table"], index=False)
    joined.to_csv(outputs["joined_table"], index=False)
    correlations.to_csv(outputs["correlation_table"], index=False)
    write_report(correlations, joined, outputs, log_path)
    logging.info(f"[PHASE] total: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
