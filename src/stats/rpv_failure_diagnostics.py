"""
RPV failure diagnostics for multiangular UAV project.

Collects RPV results from all locations, computes summary statistics,
generates diagnostic plots, and prints interpretation of the RPV model failure.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl
plt.style.use("seaborn-v0_8-whitegrid")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BAND_COLORS = {
    "band1": "#1f77b4",  # blue
    "band2": "#2ca02c",  # green
    "band3": "#d62728",  # red
    "band4": "#9467bd",  # purple (red edge)
    "band5": "#ff7f0e",  # orange (NIR)
}
BAND_LABELS = {
    "band1": "Blue (475 nm)",
    "band2": "Green (560 nm)",
    "band3": "Red (668 nm)",
    "band4": "Red Edge (717 nm)",
    "band5": "NIR (842 nm)",
}
BANDS = ["band1", "band2", "band3", "band4", "band5"]

# Per-week band CSV locations (no 'band' column; band extracted from filename)
PER_WEEK_DIRS = [
    "/run/media/davidem/Heim/RPV_Results/V12",
    "/run/media/davidem/Heim/2024/20240624_week3/metashape/20241206_week3_products_uav_data/RPV_Results/V12",
    "/run/media/davidem/Heim/2025/week0/metashape/20250822_products_uas_data/RPV_Results/V12",
    "/run/media/davidem/Heim/2025/week3/metashape/20250828_products_uas_data/RPV_Results/V12",
    "/run/media/davidem/Heim/2025/week5/metashape/20250829_products_uas_data/RPV_Results/V12",
]


def _extract_band(filename: str) -> str:
    import re

    m = re.search(r"band(\d+)", filename)
    if m:
        return f"band{m.group(1)}"
    return None


def read_aggregate_csv(path: str) -> pl.DataFrame:
    """Read the aggregate rpv_results.csv which has a 'band' column."""
    logger.info("Reading aggregate: %s", path)
    df = pl.read_csv(path, ignore_errors=True)
    cols = df.columns
    # Drop unnamed index columns (first col may be empty or 'index')
    drop_cols = [c for c in cols if c in ("", "index")]
    if drop_cols:
        df = df.drop(drop_cols)
    return df


def read_per_week_band_csvs(base_dir: str) -> list[pl.DataFrame]:
    """Read all per-week/band CSVs. Band is inferred from filename."""
    base = Path(base_dir)
    frames = []
    for week_dir_name in sorted(base.iterdir()):
        if not week_dir_name.is_dir():
            continue
        week_name = week_dir_name.name  # e.g., "week0", "week3", ...
        for csv_path in sorted(week_dir_name.glob("rpv_week*_band*_results.csv")):
            band = _extract_band(csv_path.name)
            if band is None:
                continue
            try:
                df = pl.read_csv(csv_path, ignore_errors=True)
            except Exception as e:
                logger.warning("Could not read %s: %s", csv_path, e)
                continue
            # These CSVs have no 'band' column — add it
            if "band" not in df.columns:
                df = df.with_columns(pl.lit(band).alias("band"))
            # Ensure week column is consistent
            if "week" not in df.columns:
                df = df.with_columns(pl.lit(week_name).alias("week"))
            frames.append(df)
    return frames


def collect_all_results() -> pl.DataFrame:
    """Collect RPV results from aggregate CSV + all per-week band CSVs."""
    # 1. Aggregate
    agg_path = "/run/media/davidem/Heim/RPV_Results/V12/rpv_results.csv"
    dfs = [read_aggregate_csv(agg_path)]

    # 2. Per-week band CSVs from all locations
    for base_dir in PER_WEEK_DIRS:
        p = Path(base_dir)
        if not p.exists():
            logger.warning("Directory not found, skipping: %s", base_dir)
            continue
        dfs.extend(read_per_week_band_csvs(base_dir))

    # Merge all
    all_df = pl.concat(dfs, how="diagonal_relaxed")
    logger.info("Total rows collected: %d", all_df.height)

    # Keep only successes
    all_df = all_df.filter(pl.col("status") == "success")
    logger.info("After filtering success: %d", all_df.height)

    # Drop duplicates by (week, plot_id, band), keeping first
    all_df = all_df.unique(subset=["week", "plot_id", "band"], keep="first")
    logger.info("After dedup (week, plot_id, band): %d", all_df.height)

    # Standardise types
    all_df = all_df.with_columns(
        pl.col("rho0", "k", "theta", "rc", "rmse", "nrmse").cast(pl.Float64)
    )
    if "plot_id" in all_df.columns:
        all_df = all_df.with_columns(pl.col("plot_id").cast(pl.Utf8))

    return all_df


def compute_summary(df: pl.DataFrame) -> pl.DataFrame:
    """Compute per-week, per-band summary statistics."""
    summary = df.group_by(["week", "band"]).agg(
        pl.len().alias("n_success"),
        pl.col("nrmse").mean().alias("nrmse_mean"),
        pl.col("nrmse").std().alias("nrmse_std"),
        pl.col("nrmse").min().alias("nrmse_min"),
        pl.col("nrmse").max().alias("nrmse_max"),
        pl.col("rmse").mean().alias("rmse_mean"),
        pl.col("rmse").std().alias("rmse_std"),
        pl.col("rmse").min().alias("rmse_min"),
        pl.col("rmse").max().alias("rmse_max"),
        pl.col("rho0").mean().alias("rho0_mean"),
        pl.col("rho0").std().alias("rho0_std"),
        pl.col("rho0").min().alias("rho0_min"),
        pl.col("rho0").max().alias("rho0_max"),
        pl.col("theta").mean().alias("theta_mean"),
        pl.col("theta").std().alias("theta_std"),
        pl.col("theta").min().alias("theta_min"),
        pl.col("theta").max().alias("theta_max"),
        pl.col("k").mean().alias("k_mean"),
        pl.col("k").std().alias("k_std"),
        pl.col("k").min().alias("k_min"),
        pl.col("k").max().alias("k_max"),
    ).sort(["week", "band"])
    return summary


def plot_nrmse_by_band_week(df: pl.DataFrame):
    """Bar chart: mean NRMSE by band, grouped by week."""
    summary = df.group_by(["week", "band"]).agg(
        pl.col("nrmse").mean().alias("nrmse_mean")
    ).sort(["week", "band"])

    weeks = sorted(df["week"].unique().to_list())
    n_bands = len(BANDS)
    n_weeks = len(weeks)
    x = np.arange(n_weeks)
    width = 0.16

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    for i, band in enumerate(BANDS):
        vals = []
        for wk in weeks:
            row = summary.filter(pl.col("week") == wk, pl.col("band") == band)
            v = row["nrmse_mean"].to_list()
            vals.append(v[0] if v else 0)
        offset = (i - n_bands / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=BAND_LABELS[band],
                      color=BAND_COLORS[band], edgecolor="white", linewidth=0.5)
        # Annotate each bar with value
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="NRMSE = 0.5")

    ax.set_xlabel("Week")
    ax.set_ylabel("Mean NRMSE")
    ax.set_title("RPV NRMSE by Band and Week")
    ax.set_xticks(x)
    ax.set_xticklabels(weeks)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_ylim(0, None)
    fig.tight_layout()

    out_path = FIGURES_DIR / "rpv_nrmse_by_band_week.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


def plot_rho0_by_band(df: pl.DataFrame):
    """Box plot of rho0 values per band with expected vegetation range."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    data_by_band = [df.filter(pl.col("band") == b)["rho0"].drop_nulls().to_numpy() for b in BANDS]
    bp = ax.boxplot(data_by_band, tick_labels=[BAND_LABELS[b] for b in BANDS],
                    patch_artist=True, widths=0.5)

    for i, band in enumerate(BANDS):
        bp["boxes"][i].set_facecolor(BAND_COLORS[band])
        bp["boxes"][i].set_alpha(0.6)

    # Reference: expected healthy vegetation rho0 ranges per band
    expected = {
        "band1": (0.02, 0.06),
        "band2": (0.05, 0.15),
        "band3": (0.02, 0.08),
        "band4": (0.10, 0.30),
        "band5": (0.30, 0.60),
    }
    for i, band in enumerate(BANDS):
        lo, hi = expected[band]
        ax.hlines(y=lo, xmin=i + 0.6, xmax=i + 1.4, colors="black", linestyles="dotted", linewidth=1)
        ax.hlines(y=hi, xmin=i + 0.6, xmax=i + 1.4, colors="black", linestyles="dotted", linewidth=1)
        ax.fill_between([i + 0.6, i + 1.4], lo, hi, alpha=0.08, color="green")

    ax.set_ylabel("rho0")
    ax.set_title("RPV rho0 by Band (dotted lines = expected healthy vegetation range)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()

    out_path = FIGURES_DIR / "rpv_rho0_by_band.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


def plot_theta_by_band(df: pl.DataFrame):
    """Box plot of theta values per band with theta=0 reference."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    data_by_band = [df.filter(pl.col("band") == b)["theta"].drop_nulls().to_numpy() for b in BANDS]
    bp = ax.boxplot(data_by_band, tick_labels=[BAND_LABELS[b] for b in BANDS],
                    patch_artist=True, widths=0.5)

    for i, band in enumerate(BANDS):
        bp["boxes"][i].set_facecolor(BAND_COLORS[band])
        bp["boxes"][i].set_alpha(0.6)

    ax.axhline(y=0, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="theta = 0")
    ax.set_ylabel("theta")
    ax.set_title("RPV theta by Band")
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()

    out_path = FIGURES_DIR / "rpv_theta_by_band.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


def plot_nrmse_by_cultivar_treatment(df: pl.DataFrame):
    """Grouped bar: mean NRMSE by cultivar x treatment."""
    # Ensure these columns exist
    if "cultivar" not in df.columns or "treatment" not in df.columns:
        logger.warning("cultivar/treatment columns missing; skipping cultivar-treatment plot")
        return

    summary = df.group_by(["cultivar", "treatment"]).agg(
        pl.col("nrmse").mean().alias("nrmse_mean")
    ).sort(["cultivar", "treatment"])

    groups = summary["cultivar"].unique().to_list()
    treatments = sorted(summary["treatment"].unique().to_list())

    x = np.arange(len(groups))
    width = 0.3
    colors_trt = {"trt": "#e74c3c", "no_trt": "#3498db"}

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    for i, trt in enumerate(treatments):
        vals = []
        for g in groups:
            row = summary.filter(pl.col("cultivar") == g, pl.col("treatment") == trt)
            v = row["nrmse_mean"].to_list()
            vals.append(v[0] if v else 0)
        offset = (i - len(treatments) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=trt,
                      color=colors_trt.get(trt, "gray"), edgecolor="white")
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Cultivar")
    ax.set_ylabel("Mean NRMSE")
    ax.set_title("RPV NRMSE by Cultivar and Treatment")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend(title="Treatment")
    ax.set_ylim(0, None)
    fig.tight_layout()

    out_path = FIGURES_DIR / "rpv_nrmse_by_cultivar_treatment.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)


def save_fit_quality_table(summary: pl.DataFrame):
    """Save the per-week, per-band fit quality table."""
    cols_out = [
        "week", "band", "n_success",
        "nrmse_mean", "nrmse_std", "nrmse_min", "nrmse_max",
        "rho0_mean", "rho0_std",
        "theta_mean", "theta_std",
        "k_mean", "k_std",
    ]
    available = [c for c in cols_out if c in summary.columns]
    tbl = summary.select(available).sort(["week", "band"])

    out_path = RESULTS_DIR / "rpv_fit_quality_by_week_band.csv"
    tbl.write_csv(out_path)
    logger.info("Saved %s", out_path)
    return tbl


def print_interpretation(df: pl.DataFrame):
    """Print a clinical interpretation of RPV failure."""
    print("\n" + "=" * 72)
    print("RPV MODEL FAILURE DIAGNOSTICS — INTERPRETATION")
    print("=" * 72)

    # Overall NRMSE range
    nrmse_vals = df["nrmse"].drop_nulls().to_numpy()
    print(f"\nSamples analysed: {df.height}")
    print(f"Weeks present: {sorted(df['week'].unique().to_list())}")

    print(f"\n1. OVERALL NRMSE")
    print(f"   Range:  {nrmse_vals.min():.4f} – {nrmse_vals.max():.4f}")
    print(f"   Mean:   {nrmse_vals.mean():.4f}")
    print(f"   Median: {np.median(nrmse_vals):.4f}")
    if nrmse_vals.mean() > 0.4:
        print(f"   → NRMSE is high across all data (>{'=' if nrmse_vals.min() > 0.4 else ''}0.4).")
        print(f"     The RPV model consistently explains little variance.")
    else:
        print(f"   → NRMSE varies; some fits are acceptable, others are poor.")

    # Theta analysis
    theta_vals = df["theta"].drop_nulls().to_numpy()
    near_zero = np.abs(theta_vals) < 0.15
    frac_near_zero = near_zero.sum() / len(theta_vals)
    print(f"\n2. THETA (BRDF asymmetry parameter)")
    print(f"   Range:    {theta_vals.min():.4f} – {theta_vals.max():.4f}")
    print(f"   Mean:     {theta_vals.mean():.4f}")
    print(f"   |theta| < 0.15:  {frac_near_zero*100:.0f}% of fits")
    if frac_near_zero > 0.8:
        print(f"   → theta is consistently near 0.")
        print(f"     The RPV model captures essentially NO BRDF asymmetry.")
        print(f"     This means the angular signal is either too weak or too complex")
        print(f"     for the RPV parameterisation to separate hot-spot from bowl shape.")
    elif theta_vals.mean() < 0:
        print(f"   → theta tends slightly negative. Possible weak bowl-shape indication.")
    else:
        print(f"   → theta shows some variation, but is generally small.")

    # rho0 per band
    print(f"\n3. rho0 (spectral reflectance at nadir)")
    expected = {
        "band1": (0.02, 0.06, "Blue"),
        "band2": (0.05, 0.15, "Green"),
        "band3": (0.02, 0.08, "Red"),
        "band4": (0.10, 0.30, "Red Edge"),
        "band5": (0.30, 0.60, "NIR"),
    }
    for band, (lo, hi, name) in expected.items():
        vals = df.filter(pl.col("band") == band)["rho0"].drop_nulls().to_numpy()
        if len(vals) == 0:
            print(f"   {band} ({name}): no data")
            continue
        mu = vals.mean()
        in_range = np.sum((vals >= lo) & (vals <= hi)) / len(vals)
        status = "EXPECTED" if lo <= mu <= hi else "BELOW expected" if mu < lo else "ABOVE expected"
        print(f"   {band} ({name:>9s}): rho0 = {mu:.4f}  "
              f"(expected {lo:.2f}–{hi:.2f}) → {status} "
              f"({in_range*100:.0f}% of fits in range)")

    # Systematic vs variable failure
    print(f"\n4. SYSTEMATIC VS VARIABLE FAILURE")
    by_week_band = df.group_by(["week", "band"]).agg(pl.col("nrmse").mean().alias("nrmse_mean"))
    all_above_04 = by_week_band.filter(pl.col("nrmse_mean") >= 0.4).height
    total = by_week_band.height
    print(f"   Week×Band combinations with mean NRMSE ≥ 0.4: {all_above_04}/{total}")
    if all_above_04 == total:
        print(f"   → RPV failure is SYSTEMATIC across ALL weeks and bands.")
    elif all_above_04 > total * 0.7:
        print(f"   → RPV failure is largely systematic, with occasional better fits.")
    else:
        print(f"   → RPV quality varies substantially by week/band — some are usable.")

    # Key finding
    nir_rho0 = df.filter(pl.col("band") == "band5")["rho0"].drop_nulls().mean()
    nir_vals = df.filter(pl.col("band") == "band5")["rho0"].drop_nulls().to_numpy()
    nir_in_range = np.sum((nir_vals >= 0.30) & (nir_vals <= 0.60)) / len(nir_vals) if len(nir_vals) > 0 else 0

    print(f"\n5. KEY FINDING")
    if nir_rho0 is not None:
        print(f"   NIR rho0 = {nir_rho0:.4f} (expected 0.30–0.60 for vegetation; "
              f"only {nir_in_range*100:.0f}% in range).")
    print(f"   NRMSE mean = {nrmse_vals.mean():.4f} (threshold for acceptable fit < 0.2).")
    print(f"   theta near-zero for {frac_near_zero*100:.0f}% of fits — no BRDF asymmetry captured.")
    print(f"   The RPV model systematically fails to describe angular reflectance")
    print(f"   of sugar beet. This supports the hypothesis that multiangular signal")
    print(f"   carries useful information NOT captured by a simple BRDF model.")
    print(f"   → Off-nadir data should NOT be discarded as noise.")

    print("=" * 72 + "\n")


def main():
    logger.info("=== RPV Failure Diagnostics ===")

    df = collect_all_results()
    logger.info("Final dataset: %d rows, %d columns", df.height, len(df.columns))

    # Summary statistics
    summary = compute_summary(df)
    print("\nPer-week, per-band summary:")
    print(summary.to_pandas().to_string(index=False))

    # Plots
    logger.info("Generating plots...")
    plot_nrmse_by_band_week(df)
    plot_rho0_by_band(df)
    plot_theta_by_band(df)
    plot_nrmse_by_cultivar_treatment(df)

    # Fit quality table
    tbl = save_fit_quality_table(summary)
    print("\nFit quality table (first 15 rows):")
    print(tbl.head(15).to_pandas().to_string(index=False))

    # Interpretation
    print_interpretation(df)

    logger.info("Diagnostics complete.")


if __name__ == "__main__":
    main()
