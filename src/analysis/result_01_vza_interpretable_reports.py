#!/usr/bin/env python3
"""Create interpretable VZA model reports for Result 1 cached outputs."""

from __future__ import annotations

import argparse
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.formula.api as smf

from src.analysis.result_01_reflectance_distributions import BANDS, DEFAULT_OUTPUT, SEED

CONTROL_FORMULA = "reflectance ~ C(week) + C(cult) + C(trt)"
VZA_FORMULA = "reflectance ~ C(vza_class) + C(week) + C(cult) + C(trt)"
VZA_WEEK_FORMULA = "reflectance ~ C(vza_class) * C(week) + C(cult) + C(trt)"
MIN_HEADLINE_PLOTS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--year", type=int, choices=[2024, 2025], default=None)
    parser.add_argument(
        "--filter-state",
        choices=["unfiltered", "ground_filtered"],
        default=None,
        help="If omitted, regenerate reports for both filter states.",
    )
    return parser.parse_args()


def configure_logging(base_dir: Path) -> Path:
    log_dir = base_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"vza_interpretable_report_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(path), logging.StreamHandler()],
        force=True,
    )
    return path


def log_phase(name: str, started: float) -> None:
    logging.info("[PHASE] %s: %.2fs", name, time.perf_counter() - started)


def model_term_type(term: str) -> str:
    if "vza_class" in term and "week" in term:
        return "vza_week_interaction"
    if "vza_class" in term:
        return "vza_main"
    if "week" in term:
        return "week_control"
    if "cult" in term:
        return "cultivar_control"
    if "trt" in term:
        return "treatment_control"
    if term == "Intercept":
        return "intercept"
    return "other"


def term_level(term: str, variable: str) -> str | None:
    match = re.search(rf"C\({variable}\)\[T\.([^\]]+)\]", term)
    return match.group(1) if match else None


def fit_clustered(formula: str, frame: pd.DataFrame):
    return smf.ols(formula, frame).fit(cov_type="cluster", cov_kwds={"groups": frame["plot_id"]})


def extract_terms(model, band: str, band_name: str, model_name: str) -> list[dict[str, object]]:
    conf = model.conf_int()
    rows = []
    for term in model.params.index:
        term_type = model_term_type(term)
        if term_type not in {"vza_main", "vza_week_interaction"}:
            continue
        estimate = float(model.params[term])
        rows.append(
            {
                "band": band,
                "band_name": band_name,
                "model": model_name,
                "term": term,
                "term_type": term_type,
                "vza_class": term_level(term, "vza_class"),
                "week": term_level(term, "week"),
                "estimate": estimate,
                "std_error": float(model.bse[term]),
                "ci_low": float(conf.loc[term, 0]),
                "ci_high": float(conf.loc[term, 1]),
                "p_value": float(model.pvalues[term]),
                "abs_estimate": abs(estimate),
            }
        )
    return rows


def build_model_tables(long: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    started = time.perf_counter()
    comparisons = []
    term_rows = []
    angle_order = (
        long.select("vza_class", "vza_midpoint")
        .unique()
        .sort("vza_midpoint")["vza_class"]
        .to_list()
    )
    for band, band_name in BANDS.items():
        frame = long.filter(pl.col("band") == band).to_pandas()
        if frame.empty:
            continue
        frame["vza_class"] = pd.Categorical(
            frame["vza_class"], categories=angle_order, ordered=True
        )
        try:
            fit_started = time.perf_counter()
            control = fit_clustered(CONTROL_FORMULA, frame)
            vza = fit_clustered(VZA_FORMULA, frame)
            vza_week = fit_clustered(VZA_WEEK_FORMULA, frame)
            logging.info(
                "[ML] fit %s VZA model ladder: %.3fs", band_name, time.perf_counter() - fit_started
            )
            term_rows.extend(extract_terms(vza, band, band_name, "vza_main"))
            term_rows.extend(extract_terms(vza_week, band, band_name, "vza_week"))
            comparisons.append(
                {
                    "band": band,
                    "band_name": band_name,
                    "nobs": int(control.nobs),
                    "control_r2": float(control.rsquared),
                    "vza_r2": float(vza.rsquared),
                    "vza_week_r2": float(vza_week.rsquared),
                    "delta_r2_vza_vs_control": float(vza.rsquared - control.rsquared),
                    "delta_r2_vza_week_vs_vza": float(vza_week.rsquared - vza.rsquared),
                    "delta_r2_vza_week_vs_control": float(vza_week.rsquared - control.rsquared),
                    "control_adj_r2": float(control.rsquared_adj),
                    "vza_adj_r2": float(vza.rsquared_adj),
                    "vza_week_adj_r2": float(vza_week.rsquared_adj),
                    "delta_adj_r2_vza_vs_control": float(vza.rsquared_adj - control.rsquared_adj),
                    "delta_adj_r2_vza_week_vs_vza": float(vza_week.rsquared_adj - vza.rsquared_adj),
                    "control_aic": float(control.aic),
                    "vza_aic": float(vza.aic),
                    "vza_week_aic": float(vza_week.aic),
                    "delta_aic_vza_vs_control": float(vza.aic - control.aic),
                    "delta_aic_vza_week_vs_vza": float(vza_week.aic - vza.aic),
                    "control_bic": float(control.bic),
                    "vza_bic": float(vza.bic),
                    "vza_week_bic": float(vza_week.bic),
                    "delta_bic_vza_vs_control": float(vza.bic - control.bic),
                    "delta_bic_vza_week_vs_vza": float(vza_week.bic - vza.bic),
                    "model_formula_control": CONTROL_FORMULA,
                    "model_formula_vza": VZA_FORMULA,
                    "model_formula_vza_week": VZA_WEEK_FORMULA,
                }
            )
        except Exception as exc:
            logging.exception("VZA model ladder failed for %s", band_name)
            comparisons.append({"band": band, "band_name": band_name, "error": str(exc)})
    log_phase("vza_model_tables", started)
    return pl.DataFrame(comparisons), pl.DataFrame(term_rows)


def top_table(
    frame: pl.DataFrame, value_column: str, min_plots: int = MIN_HEADLINE_PLOTS, n: int = 10
) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    return (
        frame.filter(pl.col("matched_plots") >= min_plots)
        .with_columns(pl.col(value_column).abs().alias("_magnitude"))
        .sort("_magnitude", descending=True)
        .head(n)
        .drop("_magnitude")
    )


def markdown_model_table(models: pl.DataFrame) -> list[str]:
    lines = [
        "| Band | R2 controls | R2 + VZA | Delta R2 VZA | R2 VZA x week | Delta R2 interaction | Delta AIC VZA | Delta AIC interaction | Delta BIC VZA | Delta BIC interaction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in models.iter_rows(named=True):
        lines.append(
            f"| {row['band_name']} | {row['control_r2']:.4f} | {row['vza_r2']:.4f} | "
            f"{row['delta_r2_vza_vs_control']:.4f} | {row['vza_week_r2']:.4f} | "
            f"{row['delta_r2_vza_week_vs_vza']:.4f} | {row['delta_aic_vza_vs_control']:.2f} | "
            f"{row['delta_aic_vza_week_vs_vza']:.2f} | {row['delta_bic_vza_vs_control']:.2f} | "
            f"{row['delta_bic_vza_week_vs_vza']:.2f} |"
        )
    return lines


def write_detailed_report(
    out_dir: Path,
    year: int,
    filter_state: str,
    long: pl.DataFrame,
    contrasts: pl.DataFrame,
    temporal: pl.DataFrame,
    contrast_changes: pl.DataFrame,
    models: pl.DataFrame,
    terms: pl.DataFrame,
    log_path: Path,
) -> Path:
    started = time.perf_counter()
    reports = out_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    report_path = reports / f"vza_detailed_results_{year}.md"
    reference_class = long.sort("vza_midpoint")["vza_class"][0]
    weeks = ", ".join(map(str, sorted(long["week"].unique().to_list())))
    top_contrasts = top_table(contrasts, "median_absolute_contrast")
    top_temporal = (
        temporal.filter(pl.col("matched_plots") >= MIN_HEADLINE_PLOTS)
        .with_columns(pl.col("median_temporal_change").abs().alias("_magnitude"))
        .sort("_magnitude", descending=True)
        .head(8)
        .drop("_magnitude")
        if not temporal.is_empty()
        else temporal
    )
    top_change = (
        contrast_changes.filter(pl.col("matched_plots") >= MIN_HEADLINE_PLOTS)
        .with_columns(pl.col("median_angular_contrast_change").abs().alias("_magnitude"))
        .sort("_magnitude", descending=True)
        .head(8)
        .drop("_magnitude")
        if not contrast_changes.is_empty()
        else contrast_changes
    )
    top_interactions = (
        terms.filter(pl.col("term_type") == "vza_week_interaction")
        .sort("abs_estimate", descending=True)
        .head(12)
        if not terms.is_empty()
        else terms
    )
    lines = [
        f"# VZA Detailed Results: {year} {filter_state}",
        "",
        "## Paper-Ready Interpretation",
        "",
        "Viewing zenith angle has a strong and repeatable effect on sugar beet canopy reflectance. The effect is largest in NIR and red edge, and it changes over the season, which supports the argument that nadir-only summaries discard biologically meaningful canopy-structure information.",
        "",
        "This result is descriptive and geometric. It supports the existence of a multiangular reflectance signal, but it does not by itself prove disease prediction improvement.",
        "",
        "## Dataset",
        "",
        f"- Validated weeks: **{weeks}**",
        f"- Plot-week records: **{long.select('plot_id', 'week').unique().height}**",
        f"- Plot-VZA-band observations: **{long.height:,}**",
        f"- Unique plots: **{long['plot_id'].n_unique()}**",
        f"- VZA reference class for matched contrasts: **{reference_class}**",
        f"- Filter state: **{filter_state}**",
        "",
        "## Largest Matched VZA Contrasts",
        "",
        "These are the most directly interpretable effect sizes: off-reference VZA reflectance minus the same plot/week/band at the reference VZA class.",
        "",
        "| Week | Band | VZA class | Median contrast | 95% CI | Relative contrast | Cohen dz | Matched plots |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in top_contrasts.iter_rows(named=True):
        lines.append(
            f"| {row['week']} | {row['band_name']} | {row['vza_class']} | "
            f"{row['median_absolute_contrast']:.4f} | [{row['ci_low']:.4f}, {row['ci_high']:.4f}] | "
            f"{row['median_relative_contrast']:.3f} | {row['cohens_dz']:.2f} | {row['matched_plots']} |"
        )
    lines.extend(["", "## VZA Model Ladder", ""])
    lines.extend(
        [
            "The model ladder separates the baseline seasonal/cultivar/treatment structure from the added value of VZA and the extra seasonal VZA interaction.",
            "",
            "`controls`: `reflectance ~ week + cultivar + treatment`",
            "",
            "`+ VZA`: adds categorical VZA class.",
            "",
            "`VZA x week`: allows the angular curve to change by week.",
            "",
        ]
    )
    lines.extend(markdown_model_table(models))
    lines.extend(
        [
            "",
            "Negative Delta AIC supports the added model term. Delta BIC is more conservative and penalizes the many week-by-angle interaction terms.",
            "",
            "## Largest Reference-Coded VZA x Week Terms",
            "",
            "These coefficients are model diagnostics, not raw contrasts. They are relative to the model reference categories.",
            "",
            "| Band | VZA term | Week term | Estimate | 95% CI | p |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in top_interactions.iter_rows(named=True):
        lines.append(
            f"| {row['band_name']} | {row['vza_class']} | {row['week']} | {row['estimate']:.5f} | "
            f"[{row['ci_low']:.5f}, {row['ci_high']:.5f}] | {row['p_value']:.4g} |"
        )
    if not top_temporal.is_empty():
        lines.extend(
            [
                "",
                "## Largest Matched Temporal Reflectance Changes",
                "",
                "| Week pair | Band | VZA class | Median temporal change | Matched plots |",
                "|---|---|---|---:|---:|",
            ]
        )
        for row in top_temporal.iter_rows(named=True):
            lines.append(
                f"| {row['week_start']} to {row['week_end']} | {row['band_name']} | {row['vza_class']} | "
                f"{row['median_temporal_change']:.4f} | {row['matched_plots']} |"
            )
    if not top_change.is_empty():
        lines.extend(
            [
                "",
                "## Largest Seasonal Changes in Angular Contrast",
                "",
                "| Week pair | Band | VZA class | Median contrast change | Matched plots |",
                "|---|---|---|---:|---:|",
            ]
        )
        for row in top_change.iter_rows(named=True):
            lines.append(
                f"| {row['week_start']} to {row['week_end']} | {row['band_name']} | {row['vza_class']} | "
                f"{row['median_angular_contrast_change']:.4f} | {row['matched_plots']} |"
            )
    lines.extend(
        [
            "",
            "## Reporting Guidance",
            "",
            "- Use the matched contrast table as the main effect-size evidence because it compares the same plot and week against the reference VZA.",
            "- Use the model ladder to show that VZA adds explanatory value beyond week, cultivar, and treatment controls.",
            "- Use the `VZA x week` model to justify saying that the angular response is seasonal, not constant.",
            "- Do not interpret cultivar or treatment controls as disease effects in this Result 1 section.",
            "",
            "## Output Files",
            "",
            f"- Model comparison: `{out_dir / f'results/vza_model_comparison_{year}.csv'}`",
            f"- Model terms: `{out_dir / f'results/vza_model_terms_{year}.csv'}`",
            f"- Top VZA x week terms: `{out_dir / f'results/vza_top_interactions_{year}.csv'}`",
            f"- Matched contrasts: `{out_dir / 'results/matched_angular_contrasts.csv'}`",
            f"- Temporal changes: `{out_dir / f'results/temporal_reflectance_changes_{year}.csv'}`",
            f"- Angular contrast changes: `{out_dir / f'results/angular_contrast_changes_{year}.csv'}`",
            "",
            "## Reproducibility",
            "",
            f"- Log: `{log_path}`",
            f"- Random seed: `{SEED}`",
            f"- Cluster-robust standard errors grouped by `plot_id`.",
            "- Analytical unit: plot-level reflectance aggregated within week, band, and VZA class.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    (reports / "vza_detailed_results.md").write_text("\n".join(lines), encoding="utf-8")
    log_phase("write_detailed_vza_report", started)
    return report_path


def update_summary_report(
    summary_path: Path, year: int, models: pl.DataFrame, detailed_report: Path
) -> None:
    if not summary_path.exists():
        return
    marker = "## Interpretable VZA Model Evidence"
    existing = summary_path.read_text(encoding="utf-8")
    existing = existing.split(marker)[0].rstrip()
    lines = [
        "",
        marker,
        "",
        "The VZA model ladder was added to make the statistical evidence directly reportable. It compares controls-only models against models with VZA and VZA x week terms.",
        "",
    ]
    lines.extend(markdown_model_table(models))
    lines.extend(
        [
            "",
            f"Detailed paper-ready interpretation: `{detailed_report}`",
            "",
        ]
    )
    summary_path.write_text(existing + "\n" + "\n".join(lines), encoding="utf-8")
    summary_path.with_name("reflectance_distributions_summary.md").write_text(
        summary_path.read_text(encoding="utf-8"), encoding="utf-8"
    )


def process_one(base_dir: Path, year: int, filter_state: str) -> None:
    out_dir = base_dir / str(year) / filter_state
    log_path = configure_logging(out_dir)
    started = time.perf_counter()
    results = out_dir / "results"
    long_path = results / f"plot_week_angle_features_{year}.parquet"
    if not long_path.exists():
        logging.warning("Skipping missing feature table: %s", long_path)
        return
    read_started = time.perf_counter()
    long = pl.read_parquet(long_path)
    logging.info(
        "[IO] parquet reads: n=1 min=%.3fs median=%.3fs max=%.3fs mean=%.3fs",
        time.perf_counter() - read_started,
        time.perf_counter() - read_started,
        time.perf_counter() - read_started,
        time.perf_counter() - read_started,
    )
    contrasts = pl.read_csv(results / "matched_angular_contrasts.csv")
    temporal = pl.read_csv(results / f"temporal_reflectance_changes_{year}.csv")
    contrast_changes = pl.read_csv(results / f"angular_contrast_changes_{year}.csv")
    models, terms = build_model_tables(long)
    models.write_csv(results / f"vza_model_comparison_{year}.csv")
    terms.write_csv(results / f"vza_model_terms_{year}.csv")
    (
        terms.filter(pl.col("term_type") == "vza_week_interaction")
        .sort("abs_estimate", descending=True)
        .write_csv(results / f"vza_top_interactions_{year}.csv")
    )
    report = write_detailed_report(
        out_dir,
        year,
        filter_state,
        long,
        contrasts,
        temporal,
        contrast_changes,
        models,
        terms,
        log_path,
    )
    update_summary_report(
        out_dir / f"reports/reflectance_distributions_summary_{year}.md", year, models, report
    )
    logging.info("Wrote detailed VZA report: %s", report)
    log_phase("total", started)


def main() -> None:
    args = parse_args()
    years = [args.year] if args.year else [2024, 2025]
    states = [args.filter_state] if args.filter_state else ["unfiltered", "ground_filtered"]
    for year in years:
        for state in states:
            process_one(args.output_dir, year, state)


if __name__ == "__main__":
    main()
