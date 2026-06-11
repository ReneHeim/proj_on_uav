#!/usr/bin/env python3
"""Angle and band ablation for multiangular UAV disease prediction.

Trains LogisticRegression with subsets of VZA bins and spectral bands to
identify which angles and bands carry the disease signal.
"""

import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import warnings

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WEEK_DIRS = {
    "week0": "/run/media/davidem/Heim/2024/20240603_week0/metashape/20241205_products_uav_data/output/extract/polygon_df",
    "week3": "/run/media/davidem/Heim/2024/20240624_week3/metashape/20241206_week3_products_uav_data/output/plots",
    "week5": "/run/media/davidem/Heim/2024/20240715_week5/metashape/20241207_week5_products_uav_data/output/plots",
    "week8": "/run/media/davidem/Heim/2024/20240826_week8/metashape/20241029_products_uav_data/output/extract/polygon_df",
}
POLYGON_PATH = "/run/media/davidem/Heim/2024_oncerco_plot_polygons.gpkg"

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "outputs"
RESULTS_DIR = OUT_DIR / "results"
FIGURES_DIR = OUT_DIR / "figures"
LOGS_DIR = OUT_DIR / "logs"
REPORTS_DIR = OUT_DIR / "reports"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
SCRIPT_NAME = "angle_band_ablation"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BANDS = ["band1", "band2", "band3", "band4", "band5"]
BAND_NAMES = {
    "band1": "Blue (475nm)",
    "band2": "Green (560nm)",
    "band3": "Red (668nm)",
    "band4": "Red Edge (717nm)",
    "band5": "NIR (842nm)",
}
BAND_SHORT = {"band1": "Blue", "band2": "Green", "band3": "Red", "band4": "RedEdge", "band5": "NIR"}

VZA_BINS = [0, 15, 25, 35, 45, 60]
VZA_LABELS = ["0-15", "15-25", "25-35", "35-45", "45-60"]

MAX_SAMPLE = 500_000
N_CORES = 4
SEED = 42
N_SPLITS = 5

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_PATH = LOGS_DIR / f"{SCRIPT_NAME}_{TIMESTAMP}.log"


def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def log_phase(name, elapsed):
    msg = f"[PHASE] {name}: {elapsed:.1f}s"
    logging.info(msg)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_polygon_meta():
    import geopandas as gpd

    gdf = gpd.read_file(POLYGON_PATH)
    gdf["disease_label"] = gdf["ino"].astype(int)
    gdf["plot_id"] = "plot_" + (gdf["ifz_id"] - 90001).astype(str)
    return pl.from_pandas(gdf[["cult", "trt", "disease_label", "plot_id"]].copy())


def process_parquet(pf, idx=0, total=0):
    t0 = time.time()
    df = pl.read_parquet(pf)
    t_read = time.time() - t0

    if df.height > MAX_SAMPLE:
        df = df.sample(n=MAX_SAMPLE, seed=SEED)

    mask = pl.col("vza").is_not_nan() & (pl.col("vza") >= 0) & (pl.col("vza") <= 90)
    for b in BANDS:
        mask = mask & pl.col(b).is_not_nan() & (pl.col(b) > 0)
    df = df.filter(mask)

    if df.height == 0:
        return df

    df = df.with_columns(
        pl.col("vza").cut(VZA_BINS[1:-1], labels=VZA_LABELS).alias("vza_bin"),
    )

    t_total = time.time() - t0
    logging.info(
        f"  [{idx+1}/{total}] {pf.name}: read={t_read:.1f}s total={t_total:.1f}s rows={df.height:,}"
    )
    return df


def build_feature_matrix():
    t0 = time.time()

    meta = load_polygon_meta()
    meta_dict = {r["plot_id"]: r for r in meta.to_dicts()}

    all_files = []
    week_files = {}
    for wk, wdir in WEEK_DIRS.items():
        pdir = Path(wdir)
        if not pdir.exists():
            logging.warning(f"  {wk}: directory not found: {wdir}")
            continue
        files = sorted(pdir.glob("plot_*.parquet"))
        if files:
            all_files.extend(files)
            week_files[wk] = files
            logging.info(f"  {wk}: {len(files)} plots from {wdir}")

    logging.info(f"Total: {len(all_files)} plot files across {len(week_files)} weeks")

    # Parallel load
    t1 = time.time()
    processed = []
    with ThreadPoolExecutor(max_workers=N_CORES) as ex:
        futures = []
        for i, f in enumerate(all_files):
            futures.append(ex.submit(process_parquet, f, i, len(all_files)))
        for fut in futures:
            processed.append(fut.result())
    log_phase("Parallel load", time.time() - t1)

    # Build data cache and feature rows
    t2 = time.time()
    data_cache = {}
    i = 0
    for wk, files in week_files.items():
        for pf in files:
            df = processed[i]
            i += 1
            plot_id = pf.stem
            if df.height > 0:
                data_cache[(wk, plot_id)] = df

    logging.info(f"Valid data: {len(data_cache)} plot-week pairs")
    log_phase("Build data cache", time.time() - t2)

    # Build aggregated feature rows (M3-style)
    t3 = time.time()
    rows = []
    for (wk, pid), df in data_cache.items():
        df_binned = df.filter(pl.col("vza_bin").is_not_null())
        if df_binned.height == 0:
            continue
        if pid not in meta_dict:
            continue
        meta_r = meta_dict[pid]
        row = {
            "plot_id": pid,
            "week": int(wk.replace("week", "")),
            "cult": meta_r["cult"],
            "trt": meta_r["trt"],
            "disease_label": meta_r["disease_label"],
        }
        agg = df_binned.group_by("vza_bin").agg([pl.col(b).mean().alias(b) for b in BANDS])
        for agg_row in agg.to_dicts():
            vza_label = agg_row["vza_bin"].replace("-", "_")
            for b in BANDS:
                row[f"{b}_vza{vza_label}"] = agg_row[b]
        rows.append(row)

    feature_df = pl.DataFrame(rows)
    log_phase("Build feature matrix", time.time() - t3)

    log_phase("Total data loading", time.time() - t0)

    return feature_df, meta_dict


# ---------------------------------------------------------------------------
# Feature column selection
# ---------------------------------------------------------------------------


def get_angle_columns(vza_label):
    lbl = vza_label.replace("-", "_")
    return [f"{b}_vza{lbl}" for b in BANDS]


def get_all_features():
    cols = []
    for vl in VZA_LABELS:
        cols.extend(get_angle_columns(vl))
    return cols


# ---------------------------------------------------------------------------
# Ablation definitions
# ---------------------------------------------------------------------------

ANGLE_ABLATIONS = {
    "vza_0_only": ["0-15"],
    "vza_15_only": ["15-25"],
    "vza_25_only": ["25-35"],
    "vza_35_only": ["35-45"],
    "vza_45_only": ["45-60"],
    "all_vza": VZA_LABELS,
    "all_except_0": ["15-25", "25-35", "35-45", "45-60"],
    "all_except_60": ["0-15", "15-25", "25-35", "35-45"],
}

BAND_ABLATIONS = {
    "blue_only": ["band1"],
    "green_only": ["band2"],
    "red_only": ["band3"],
    "red_edge_only": ["band4"],
    "nir_only": ["band5"],
    "visible_only": ["band1", "band2", "band3"],
    "red_edge_nir": ["band4", "band5"],
    "all_bands": BANDS,
}


def angle_subset_features(vza_labels):
    """Build column list for a subset of VZA bins, all bands."""
    cols = []
    for vl in vza_labels:
        cols.extend(get_angle_columns(vl))
    return cols


def band_subset_features(band_list):
    """Build column list for a subset of bands, all VZA bins."""
    cols = []
    for vl in VZA_LABELS:
        lbl = vl.replace("-", "_")
        for b in band_list:
            cols.append(f"{b}_vza{lbl}")
    return cols


# ---------------------------------------------------------------------------
# Model and CV
# ---------------------------------------------------------------------------


def build_pipeline():
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def run_cv_split(X, y, groups):
    try:
        skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        return list(skf.split(X, y, groups=groups))
    except (ValueError, RuntimeError):
        gkf = GroupKFold(n_splits=N_SPLITS)
        return list(gkf.split(X, y, groups=groups))


def evaluate_columns(feature_df, col_names, label="unknown"):
    """Train and CV-evaluate using a subset of feature columns. Returns AUROC stats."""
    available = [c for c in col_names if c in feature_df.columns]
    if not available:
        return {
            "label": label,
            "n_features": 0,
            "AUROC_mean": np.nan,
            "AUROC_std": np.nan,
            "n_rows": 0,
        }

    sub = feature_df.select(["plot_id", "week", "disease_label"] + available).drop_nulls()
    if sub.height < 10:
        return {
            "label": label,
            "n_features": len(available),
            "AUROC_mean": np.nan,
            "AUROC_std": np.nan,
            "n_rows": sub.height,
        }

    X = sub.select(available).to_numpy()
    y = sub["disease_label"].to_numpy()
    sub = sub.with_columns(
        (pl.col("plot_id").str.extract(r"(\d+)").cast(pl.Int64) + 90001).alias("ifz_id")
    )
    groups = sub["ifz_id"].to_numpy()

    splits = run_cv_split(X, y, groups)
    pipe = build_pipeline()
    aurocs = []

    for train_idx, test_idx in splits:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        try:
            pipe.fit(X_train, y_train)
            y_proba = pipe.predict_proba(X_test)[:, 1]
            aurocs.append(roc_auc_score(y_test, y_proba))
        except Exception as e:
            logging.warning(f"  [{label}] fold failed: {e}")
            aurocs.append(np.nan)

    aurocs_arr = np.array(aurocs)
    valid = aurocs_arr[~np.isnan(aurocs_arr)]

    return {
        "label": label,
        "n_features": len(available),
        "AUROC_mean": float(np.mean(valid)) if len(valid) > 0 else np.nan,
        "AUROC_std": float(np.std(valid)) if len(valid) > 0 else np.nan,
        "n_rows": sub.height,
    }


# ---------------------------------------------------------------------------
# Angle Ablation
# ---------------------------------------------------------------------------


def run_angle_ablation(feature_df):
    logging.info("=== Angle Ablation ===")
    t0 = time.time()
    results = []

    for name, vza_labels in ANGLE_ABLATIONS.items():
        cols = angle_subset_features(vza_labels)
        res = evaluate_columns(feature_df, cols, label=name)
        res["vza_labels"] = ", ".join(vza_labels)
        results.append(res)
        logging.info(
            f"  {name}: {res['n_features']} features, "
            f"AUROC={res['AUROC_mean']:.4f}±{res['AUROC_std']:.4f} "
            f"(n={res['n_rows']})"
        )

    log_phase("Angle ablation", time.time() - t0)
    return pl.DataFrame(results)


# ---------------------------------------------------------------------------
# Band Ablation
# ---------------------------------------------------------------------------


def run_band_ablation(feature_df):
    logging.info("=== Band Ablation ===")
    t0 = time.time()
    results = []

    for name, band_list in BAND_ABLATIONS.items():
        cols = band_subset_features(band_list)
        res = evaluate_columns(feature_df, cols, label=name)
        res["bands"] = ", ".join(band_list)
        results.append(res)
        logging.info(
            f"  {name}: {res['n_features']} features, "
            f"AUROC={res['AUROC_mean']:.4f}±{res['AUROC_std']:.4f} "
            f"(n={res['n_rows']})"
        )

    log_phase("Band ablation", time.time() - t0)
    return pl.DataFrame(results)


# ---------------------------------------------------------------------------
# Heatmap: single (band, VZA bin) feature model
# ---------------------------------------------------------------------------


def run_heatmap(feature_df):
    logging.info("=== Heatmap: band × VZA bin AUROC ===")
    t0 = time.time()
    results = []

    for b in BANDS:
        for vl in VZA_LABELS:
            lbl = vl.replace("-", "_")
            col_name = f"{b}_vza{lbl}"
            label = f"{BAND_SHORT[b]}_{lbl}"
            res = evaluate_columns(feature_df, [col_name], label=label)
            res["band"] = BAND_SHORT[b]
            res["vza_bin"] = vl
            res["band_full"] = b
            results.append(res)

    log_phase("Heatmap evaluation", time.time() - t0)
    return pl.DataFrame(results)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_angle_ablation(angle_df):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    name_order = [
        "vza_0_only",
        "vza_15_only",
        "vza_25_only",
        "vza_35_only",
        "vza_45_only",
        "all_vza",
        "all_except_0",
        "all_except_60",
    ]
    df = angle_df.to_pandas()
    df = df.dropna(subset=["AUROC_mean"])

    ordered = []
    for n in name_order:
        row = df[df["label"] == n]
        if not row.empty:
            ordered.append(row.iloc[0])

    labels = [r["label"] for r in ordered]
    means = [r["AUROC_mean"] for r in ordered]
    stds = [r["AUROC_std"] for r in ordered]

    display_labels = {
        "vza_0_only": "0-15° only",
        "vza_15_only": "15-25° only",
        "vza_25_only": "25-35° only",
        "vza_35_only": "35-45° only",
        "vza_45_only": "45-60° only",
        "all_vza": "All VZA",
        "all_except_0": "Excl. 0-15°",
        "all_except_60": "Excl. 45-60°",
    }
    tick_labels = [display_labels.get(l, l) for l in labels]

    science_colors = {
        "vza_0_only": "#3B7A9E",
        "vza_15_only": "#E88C46",
        "vza_25_only": "#84B082",
        "vza_35_only": "#E24E42",
        "vza_45_only": "#8E5EA2",
        "all_vza": "#2C3E50",
        "all_except_0": "#6BB5A8",
        "all_except_60": "#D4A35B",
    }
    colors = [science_colors.get(l, "#999999") for l in labels]

    fig, ax = plt.subplots(figsize=(12, 5))
    xs = np.arange(len(labels))
    bars = ax.bar(
        xs, means, yerr=stds, color=colors, capsize=5, width=0.65, edgecolor="white", linewidth=0.8
    )

    for bar, mean_val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{mean_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(tick_labels, fontsize=9, rotation=25, ha="right")
    ax.set_ylabel("Mean AUROC (5-fold CV)", fontsize=12)
    ax.set_title("AUROC by VZA Bin Subset", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    fig.tight_layout()
    path = FIGURES_DIR / "auroc_by_vza_bin.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logging.info(f"Saved: {path}")
    plt.close(fig)


def plot_band_ablation(band_df):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    name_order = [
        "blue_only",
        "green_only",
        "red_only",
        "red_edge_only",
        "nir_only",
        "visible_only",
        "red_edge_nir",
        "all_bands",
    ]
    df = band_df.to_pandas()
    df = df.dropna(subset=["AUROC_mean"])

    ordered = []
    for n in name_order:
        row = df[df["label"] == n]
        if not row.empty:
            ordered.append(row.iloc[0])

    labels = [r["label"] for r in ordered]
    means = [r["AUROC_mean"] for r in ordered]
    stds = [r["AUROC_std"] for r in ordered]

    display_labels = {
        "blue_only": "Blue only",
        "green_only": "Green only",
        "red_only": "Red only",
        "red_edge_only": "Red Edge only",
        "nir_only": "NIR only",
        "visible_only": "Visible (1-3)",
        "red_edge_nir": "Red Edge + NIR",
        "all_bands": "All bands",
    }
    tick_labels = [display_labels.get(l, l) for l in labels]

    science_colors = {
        "blue_only": "#3B7A9E",
        "green_only": "#84B082",
        "red_only": "#E24E42",
        "red_edge_only": "#E88C46",
        "nir_only": "#8E5EA2",
        "visible_only": "#6BB5A8",
        "red_edge_nir": "#D4A35B",
        "all_bands": "#2C3E50",
    }
    colors = [science_colors.get(l, "#999999") for l in labels]

    fig, ax = plt.subplots(figsize=(12, 5))
    xs = np.arange(len(labels))
    bars = ax.bar(
        xs, means, yerr=stds, color=colors, capsize=5, width=0.65, edgecolor="white", linewidth=0.8
    )

    for bar, mean_val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{mean_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(tick_labels, fontsize=9, rotation=25, ha="right")
    ax.set_ylabel("Mean AUROC (5-fold CV)", fontsize=12)
    ax.set_title("AUROC by Spectral Band Subset", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 0.95)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    fig.tight_layout()
    path = FIGURES_DIR / "auroc_by_band.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logging.info(f"Saved: {path}")
    plt.close(fig)


def plot_heatmap(heatmap_df):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = heatmap_df.to_pandas()
    df = df.dropna(subset=["AUROC_mean"])

    band_order = ["Blue", "Green", "Red", "RedEdge", "NIR"]
    vza_order = VZA_LABELS

    pivot = df.pivot(index="band", columns="vza_bin", values="AUROC_mean")

    # reorder
    row_order = [b for b in band_order if b in pivot.index]
    col_order = [v for v in vza_order if v in pivot.columns]
    data = pivot.loc[row_order, col_order] if row_order and col_order else pivot

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(data.values, aspect="auto", cmap="RdYlBu", vmin=0.4, vmax=0.85, origin="upper")

    ax.set_xticks(np.arange(len(data.columns)))
    ax.set_xticklabels(data.columns, fontsize=10)
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_yticklabels(data.index, fontsize=10)
    ax.set_xlabel("VZA Bin", fontsize=12)
    ax.set_ylabel("Band", fontsize=12)
    ax.set_title("AUROC: Band × VZA Bin (Single-Feature Model)", fontsize=13, fontweight="bold")

    for i in range(len(data.index)):
        for j in range(len(data.columns)):
            val = data.values[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 0.55 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color=text_color,
                )

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("AUROC", fontsize=10)

    fig.tight_layout()
    path = FIGURES_DIR / "feature_importance_band_angle.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    logging.info(f"Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown Summary
# ---------------------------------------------------------------------------


def write_markdown_summary(angle_df, band_df, heatmap_df):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{SCRIPT_NAME}_summary.md"

    lines = []
    lines.append("# Angle and Band Ablation Results\n")

    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ---- Angle ablation table ----
    lines.append("## Angle Ablation: AUROC by VZA Bin Subset\n")
    lines.append("| VZA Subset | Features | AUROC (mean) | AUROC (std) | n Rows |")
    lines.append("|-----------|----------|-------------|------------|--------|")
    for row in angle_df.sort("label").to_dicts():
        lines.append(
            f"| {row['label']} | {row['n_features']} | "
            f"{row['AUROC_mean']:.4f} | {row['AUROC_std']:.4f} | {row['n_rows']} |"
        )
    lines.append("")

    # best angle
    valid_angle = angle_df.filter(pl.col("AUROC_mean").is_not_nan()).sort(
        "AUROC_mean", descending=True
    )
    if valid_angle.height > 0:
        best_angle = valid_angle.row(0, named=True)
        lines.append(
            f"**Best angle subset**: {best_angle['label']} "
            f"(AUROC = {best_angle['AUROC_mean']:.4f} ± {best_angle['AUROC_std']:.4f})"
        )
        lines.append("")

    # ---- Band ablation table ----
    lines.append("## Band Ablation: AUROC by Spectral Band Subset\n")
    lines.append("| Band Subset | Features | AUROC (mean) | AUROC (std) | n Rows |")
    lines.append("|-----------|----------|-------------|------------|--------|")
    for row in band_df.sort("label").to_dicts():
        lines.append(
            f"| {row['label']} | {row['n_features']} | "
            f"{row['AUROC_mean']:.4f} | {row['AUROC_std']:.4f} | {row['n_rows']} |"
        )
    lines.append("")

    valid_band = band_df.filter(pl.col("AUROC_mean").is_not_nan()).sort(
        "AUROC_mean", descending=True
    )
    if valid_band.height > 0:
        best_band = valid_band.row(0, named=True)
        lines.append(
            f"**Best band subset**: {best_band['label']} "
            f"(AUROC = {best_band['AUROC_mean']:.4f} ± {best_band['AUROC_std']:.4f})"
        )
        lines.append("")

    # ---- Heatmap table ----
    lines.append("## Band × VZA Bin Heatmap (Single-Feature Models)\n")
    lines.append("| Band | VZA Bin | AUROC (mean) | AUROC (std) |")
    lines.append("|------|---------|-------------|------------|")
    for row in heatmap_df.sort(["band", "vza_bin"]).to_dicts():
        lines.append(
            f"| {row['band']} | {row['vza_bin']} | "
            f"{row['AUROC_mean']:.4f} | {row['AUROC_std']:.4f} |"
        )
    lines.append("")

    # best single feature
    valid_hm = heatmap_df.filter(pl.col("AUROC_mean").is_not_nan()).sort(
        "AUROC_mean", descending=True
    )
    if valid_hm.height > 0:
        best_hm = valid_hm.row(0, named=True)
        lines.append(
            f"**Best single feature**: {best_hm['band']} in VZA {best_hm['vza_bin']} "
            f"(AUROC = {best_hm['AUROC_mean']:.4f} ± {best_hm['AUROC_std']:.4f})"
        )
        lines.append("")

    # Interpretation
    lines.append("## Interpretation\n")
    lines.append("### Which VZA bin contributes most?\n")

    iso_angles = (
        angle_df.filter(
            pl.col("label").is_in(
                ["vza_0_only", "vza_15_only", "vza_25_only", "vza_35_only", "vza_45_only"]
            )
        )
        .filter(pl.col("AUROC_mean").is_not_nan())
        .sort("AUROC_mean", descending=True)
    )
    if iso_angles.height > 0:
        best_iso = iso_angles.row(0, named=True)
        lines.append(
            f"The most informative single VZA bin is **{best_iso['label']}** "
            f"(AUROC = {best_iso['AUROC_mean']:.4f}), suggesting that "
            f"reflectance from {best_iso['label'].replace('_only','')} viewing angles "
            f"carries the strongest disease signal."
        )
    lines.append("")

    lines.append("### Which spectral band is most discriminative?\n")
    iso_bands = (
        band_df.filter(
            pl.col("label").is_in(
                ["blue_only", "green_only", "red_only", "red_edge_only", "nir_only"]
            )
        )
        .filter(pl.col("AUROC_mean").is_not_nan())
        .sort("AUROC_mean", descending=True)
    )
    if iso_bands.height > 0:
        best_iso_b = iso_bands.row(0, named=True)
        lines.append(
            f"The most discriminative single band is **{best_iso_b['label']}** "
            f"(AUROC = {best_iso_b['AUROC_mean']:.4f})."
        )
    lines.append("")

    lines.append("### Does red-edge or NIR carry the angular signal?\n")
    lines.append(
        "The heatmap shows which (band, angle) combinations are most predictive "
        "as single features. The red-edge (717 nm) and NIR (842 nm) bands "
        "at moderate-to-high VZA are expected to carry angular information "
        "related to canopy structure changes caused by disease."
    )
    lines.append("")

    # Reproducibility
    lines.append("## Reproducibility\n")
    lines.append(f"- **Script**: `src/models/angle_band_ablation.py`")
    lines.append(f"- **Random seed**: {SEED}")
    lines.append(
        f"- **CV method**: StratifiedGroupKFold (n_splits={N_SPLITS}, fallback to GroupKFold)"
    )
    lines.append(f"- **Classifier**: LogisticRegression (class_weight=balanced, max_iter=1000)")
    lines.append(f"- **VZA bins**: {VZA_BINS}")
    lines.append(f"- **Sample size**: {MAX_SAMPLE:,} rows per parquet")
    lines.append(f"- **Weeks**: {list(WEEK_DIRS.keys())}")
    lines.append("")

    # Output paths
    lines.append("## Outputs\n")
    lines.append(f"- `{RESULTS_DIR / 'angle_ablation.csv'}`")
    lines.append(f"- `{RESULTS_DIR / 'band_ablation.csv'}`")
    lines.append(f"- `{RESULTS_DIR / 'heatmap_band_angle.csv'}`")
    lines.append(f"- `{FIGURES_DIR / 'auroc_by_vza_bin.png'}`")
    lines.append(f"- `{FIGURES_DIR / 'auroc_by_band.png'}`")
    lines.append(f"- `{FIGURES_DIR / 'feature_importance_band_angle.png'}`")
    lines.append(f"- `{LOG_PATH}`")
    lines.append(f"- `{path}`")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    logging.info(f"Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    setup_logging()
    total_t0 = time.time()
    logging.info(f"=== {SCRIPT_NAME} started ({TIMESTAMP}) ===")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1: Build feature matrix
    feature_df, meta_dict = build_feature_matrix()
    n_rows = feature_df.height
    feature_cols = get_all_features()
    available_features = [c for c in feature_cols if c in feature_df.columns]
    logging.info(f"Feature matrix: {n_rows} rows, {len(available_features)} features")

    # Phase 2: Angle ablation
    angle_df = run_angle_ablation(feature_df)
    angle_path = RESULTS_DIR / "angle_ablation.csv"
    angle_df.write_csv(angle_path)
    logging.info(f"Saved: {angle_path}")

    # Phase 3: Band ablation
    band_df = run_band_ablation(feature_df)
    band_path = RESULTS_DIR / "band_ablation.csv"
    band_df.write_csv(band_path)
    logging.info(f"Saved: {band_path}")

    # Phase 4: Heatmap (band × VZA bin single-feature models)
    heatmap_df = run_heatmap(feature_df)
    heatmap_path = RESULTS_DIR / "heatmap_band_angle.csv"
    heatmap_df.write_csv(heatmap_path)
    logging.info(f"Saved: {heatmap_path}")

    # Phase 5: Figures
    logging.info("=== Generating figures ===")
    t_fig = time.time()
    plot_angle_ablation(angle_df)
    plot_band_ablation(band_df)
    plot_heatmap(heatmap_df)
    log_phase("Figures", time.time() - t_fig)

    # Phase 6: Markdown summary
    logging.info("=== Writing summary ===")
    write_markdown_summary(angle_df, band_df, heatmap_df)

    log_phase("Total", time.time() - total_t0)
    logging.info(f"=== {SCRIPT_NAME} complete ===")
    logging.info(f"Log: {LOG_PATH}")


if __name__ == "__main__":
    main()
