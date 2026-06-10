#!/usr/bin/env python3
"""Build feature sets M0-M5 from per-plot parquets.
Optimized: single-pass loading, cached filters, parallel I/O.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
import polars as pl

VZA_BINS = [0, 15, 25, 35, 45, 60]
VZA_LABELS = ["0-15", "15-25", "25-35", "35-45", "45-60"]
RAA_BINS = [0, 90, 180]
RAA_LABELS = ["0-90", "90-180"]
MAX_SAMPLE = 2_000_000  # Max total pixels per file
SAMPLES_PER_BIN = 100_000  # Equal pixels per VZA bin to prevent sample imbalance bias
N_CORES = 4  # USB drive I/O bottleneck, not CPU

WEEK_DIRS = {
    # 2024 season
    "2024_wk0": "/run/media/davidem/Heim/2024/20240603_week0/metashape/20241205_products_uav_data/output/extract/polygon_df",
    "2024_wk3": "/run/media/davidem/Heim/2024/20240624_week3/metashape/20241206_week3_products_uav_data/output/plots",
    "2024_wk5": "/run/media/davidem/Heim/2024/20240715_week5/metashape/20241207_week5_products_uav_data/output/plots",
    "2024_wk8": "/run/media/davidem/Heim/2024/20240826_week8/metashape/20241029_products_uav_data/output/extract/polygon_df",
    # 2025 season
    "2025_wk0": "/run/media/davidem/Heim/2025/week0/metashape/20250822_products_uas_data/output/plots",
    "2025_wk3": "/run/media/davidem/Heim/2025/week3/metashape/20250828_products_uas_data/output/plots",
    "2025_wk5": "/run/media/davidem/Heim/2025/week5/metashape/20250829_products_uas_data/output/plots",
}
POLYGON_PATHS = {
    "2024": "/run/media/davidem/Heim/2024_oncerco_plot_polygons.gpkg",
    "2025": "/run/media/davidem/Heim/2025_oncerco_plot_polygons.gpkg",
}
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "features"


def load_polygon_meta():
    """Load polygon metadata for both seasons. Returns dict keyed by (season, plot_id)."""
    import geopandas as gpd
    
    meta = {}
    for season, path in POLYGON_PATHS.items():
        gdf = gpd.read_file(path)
        if "ino" not in gdf.columns:
            gdf["ino"] = gdf["trt"] == "no_trt"  # infer from treatment
        
        # ifz_id → zero-indexed plot_id
        for i, row in gdf.iterrows():
            if "ifz_id" in gdf.columns:
                plot_idx = row["ifz_id"] - 90001
            else:
                plot_idx = i
            pid = f"plot_{plot_idx}"
            meta[(season, pid)] = {
                "cult": row.get("cult", row.get("cultivar")),
                "trt": row["trt"],
                "disease_label": int(row["ino"]),
            }
    return meta


def load_and_preprocess(pf, idx=0, total=0):
    """Load one parquet, filter, bin, then sample equally from each VZA bin."""
    t0 = time.time()
    df = pl.read_parquet(pf)
    t_read = time.time() - t0

    # Single filter pass for all validity checks
    bands = ["band1", "band2", "band3", "band4", "band5"]
    mask = (
        pl.col("vza").is_not_nan() & (pl.col("vza") >= 0) & (pl.col("vza") <= 90)
    )
    for b in bands:
        mask = mask & pl.col(b).is_not_nan() & (pl.col(b) > 0)
    df = df.filter(mask)

    if df.height == 0:
        return df

    # Compute derived columns
    df = df.with_columns([
        ((pl.col("saa").cast(pl.Float64) - pl.col("vaa").cast(pl.Float64) + 180) % 360 - 180).abs().alias("raa"),
        (90 - pl.col("sunelev")).alias("sza"),
    ])
    df = df.with_columns([
        pl.col("vza").cut(VZA_BINS[1:-1], labels=VZA_LABELS).alias("vza_bin"),
        pl.col("raa").cast(pl.Float64).cut([90], labels=RAA_LABELS).alias("raa_bin"),
    ])

    # --- Balanced per-bin sampling ---
    # Enforce equal pixel count per VZA bin to prevent sample-imbalance bias
    balanced_parts = []
    for lbl in VZA_LABELS:
        subset = df.filter(pl.col("vza_bin") == lbl)
        if subset.height == 0:
            continue
        n_take = min(SAMPLES_PER_BIN, subset.height)
        balanced_parts.append(subset.sample(n=n_take, seed=42))
    if not balanced_parts:
        return df.clear()
    df = pl.concat(balanced_parts, how="diagonal_relaxed")

    t_total = time.time() - t0
    print(f"  [{idx+1}/{total}] {pf.name}: read={t_read:.1f}s total={t_total:.1f}s rows={df.height:,}")
    return df

    return df


def nmap(f, items):
    """Parallel map using ThreadPoolExecutor (I/O bound)."""
    results = []
    with ThreadPoolExecutor(max_workers=N_CORES) as ex:
        for r in ex.map(f, items):
            results.append(r)
    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Phase 1: Load metadata ----
    t0 = time.time()
    meta = load_polygon_meta()  # keyed by (season, plot_id)

    def get_meta(wk, pid):
        """Get metadata for a (week, plot_id) pair, handling season."""
        season = "2025" if wk.startswith("2025") else "2024"
        m = meta.get((season, pid), None)
        if m is not None:
            m = dict(m)
            m["year"] = int(season)
        return m

    # ---- Phase 2: Collect all file paths ----
    all_files = []
    week_files = {}
    for wk, wdir in WEEK_DIRS.items():
        pdir = Path(wdir)
        if not pdir.exists():
            continue
        files = sorted(pdir.glob("plot_*.parquet"))
        if files:
            all_files.extend(files)
            week_files[wk] = files
            print(f"  {wk}: {len(files)} plots from {wdir}")

    print(f"\nTotal: {len(all_files)} plot files across {len(week_files)} weeks")
    print(f"Processing with {N_CORES} workers...\n")

    # ---- Phase 3: Single-pass load + preprocess (parallel) ----
    t1 = time.time()
    fn_with_idx = partial(load_and_preprocess, idx=0, total=0)
    processed = []
    with ThreadPoolExecutor(max_workers=N_CORES) as ex:
        futures = []
        for i, f in enumerate(all_files):
            futures.append(ex.submit(load_and_preprocess, f, i, len(all_files)))
        for fut in futures:
            processed.append(fut.result())
    print(f"Loaded {len(processed)} files in {time.time() - t1:.1f}s")

    # Build lookup: (week, plot_id) → DataFrame
    data_cache = {}
    i = 0
    for wk, files in week_files.items():
        week_num = int(wk.split("_wk")[-1])
        for pf in files:
            df = processed[i]
            i += 1
            plot_id = pf.stem
            if df.height > 0:
                data_cache[(wk, plot_id)] = df

    print(f"Valid data: {len(data_cache)} plot-week pairs\n")

    # ---- Phase 4: Build all feature sets from cache ----
    t2 = time.time()
    results = {}

    # M0: metadata only
    print("=== M0: metadata ===")
    rows_m0 = []
    for wk, files in week_files.items():
        week_num = int(wk.split("_wk")[-1])
        for pf in files:
            pid = pf.stem
            mr = get_meta(wk, pid)
            if mr is not None:
                rows_m0.append({
                    "plot_id": pid, "week": week_num,
                    "cult": mr["cult"], "trt": mr["trt"],
                    "disease_label": mr["disease_label"],
                    "year": mr["year"],
                })
    m0 = pl.DataFrame(rows_m0).select(["plot_id", "week", "year", "cult", "trt", "disease_label"])
    m0.write_parquet(OUT_DIR / "M0_metadata.parquet")
    print(f"  M0: {m0.shape[0]} rows x {m0.shape[1]} cols")
    results["M0"] = m0

    bands = ["band1", "band2", "band3", "band4", "band5"]
    band_names = {b: i for i, b in enumerate(bands)}

    # M1: nadir band means
    print("=== M1: nadir bands ===")
    rows_m1 = []
    for (wk, pid), df in data_cache.items():
        df_nadir = df.filter(pl.col("vza") <= 15)
        if df_nadir.height == 0:
            continue
        meta_r = get_meta(wk, pid)
        if meta_r is None:
            continue
        row = {"plot_id": pid, "week": int(wk.split("_wk")[-1]), "cult": meta_r["cult"],
               "trt": meta_r["trt"], "disease_label": meta_r["disease_label"], "year": meta_r["year"]}
        for b in bands:
            row[f"{b}_nadir_mean"] = df_nadir[b].mean()
        rows_m1.append(row)
    m1 = pl.DataFrame(rows_m1)
    m1.write_parquet(OUT_DIR / "M1_nadir_bands.parquet")
    print(f"  M1: {m1.shape[0]} rows x {m1.shape[1]} cols")
    results["M1"] = m1

    # M2: nadir indices
    print("=== M2: nadir indices ===")
    rows_m2 = []
    for (wk, pid), df in data_cache.items():
        df_nadir = df.filter(pl.col("vza") <= 15)
        if df_nadir.height == 0:
            continue
        meta_r = get_meta(wk, pid)
        if meta_r is None:
            continue
        row = {"plot_id": pid, "week": int(wk.split("_wk")[-1]), "cult": meta_r["cult"],
               "trt": meta_r["trt"], "disease_label": meta_r["disease_label"], "year": meta_r["year"]}
        b1, b2, b3, b4, b5 = df_nadir["band1"], df_nadir["band2"], df_nadir["band3"], df_nadir["band4"], df_nadir["band5"]
        row["ndvi_nadir"] = ((b5 - b3) / (b5 + b3)).replace(np.nan, None).mean()
        row["gndvi_nadir"] = ((b5 - b2) / (b5 + b2)).replace(np.nan, None).mean()
        row["ndre_nadir"] = ((b5 - b4) / (b5 + b4)).replace(np.nan, None).mean()
        row["red_edge_ratio_nadir"] = (b4 / b2).replace(np.nan, None).mean()
        row["nir_red_ratio_nadir"] = (b5 / b3).replace(np.nan, None).mean()
        rows_m2.append(row)
    m2 = pl.DataFrame(rows_m2)
    m2.write_parquet(OUT_DIR / "M2_nadir_indices.parquet")
    print(f"  M2: {m2.shape[0]} rows x {m2.shape[1]} cols")
    results["M2"] = m2

    # M3: multiangular VZA (per-band, per-VZA-bin mean)
    print("=== M3: multiangular VZA ===")
    rows_m3 = []
    for (wk, pid), df in data_cache.items():
        df_binned = df.filter(pl.col("vza_bin").is_not_null())
        if df_binned.height == 0:
            continue
        meta_r = get_meta(wk, pid)
        if meta_r is None:
            continue
        row = {"plot_id": pid, "week": int(wk.split("_wk")[-1]), "cult": meta_r["cult"],
               "trt": meta_r["trt"], "disease_label": meta_r["disease_label"], "year": meta_r["year"]}
        agg = df_binned.group_by("vza_bin").agg([pl.col(b).mean().alias(b) for b in bands])
        for agg_row in agg.to_dicts():
            vza_label = agg_row["vza_bin"].replace("-", "_")
            for b in bands:
                row[f"{b}_vza{vza_label}"] = agg_row[b]
        rows_m3.append(row)
    m3 = pl.DataFrame(rows_m3)
    m3.write_parquet(OUT_DIR / "M3_multiangular_vza.parquet")
    print(f"  M3: {m3.shape[0]} rows x {m3.shape[1]} cols")
    results["M3"] = m3

    # M4: multiangular VZA + RAA
    print("=== M4: multiangular VZA+RAA ===")
    rows_m4 = []
    for (wk, pid), df in data_cache.items():
        df_binned = df.filter(pl.col("vza_bin").is_not_null() & pl.col("raa_bin").is_not_null())
        if df_binned.height == 0:
            continue
        meta_r = get_meta(wk, pid)
        if meta_r is None:
            continue
        row = {"plot_id": pid, "week": int(wk.split("_wk")[-1]), "cult": meta_r["cult"],
               "trt": meta_r["trt"], "disease_label": meta_r["disease_label"], "year": meta_r["year"]}
        agg = df_binned.group_by(["vza_bin", "raa_bin"]).agg([pl.col(b).mean().alias(b) for b in bands])
        for agg_row in agg.to_dicts():
            vza_label = agg_row["vza_bin"].replace("-", "_")
            raa_label = agg_row["raa_bin"].replace("-", "_")
            for b in bands:
                row[f"{b}_vza{vza_label}_raa{raa_label}"] = agg_row[b]
        rows_m4.append(row)
    m4 = pl.DataFrame(rows_m4)
    m4.write_parquet(OUT_DIR / "M4_multiangular_vza_raa.parquet")
    print(f"  M4: {m4.shape[0]} rows x {m4.shape[1]} cols")
    results["M4"] = m4

    # M5: angular contrast
    print("=== M5: angular contrast ===")
    rows_m5 = []
    for (wk, pid), df in data_cache.items():
        df_binned = df.filter(pl.col("vza_bin").is_not_null())
        if df_binned.height == 0:
            continue
        meta_r = get_meta(wk, pid)
        if meta_r is None:
            continue
        row = {"plot_id": pid, "week": int(wk.split("_wk")[-1]), "cult": meta_r["cult"],
               "trt": meta_r["trt"], "disease_label": meta_r["disease_label"], "year": meta_r["year"]}
        for b in bands:
            means = df_binned.group_by("vza_bin").agg(pl.col(b).mean()).sort("vza_bin")
            vals = {r["vza_bin"]: r[b] for r in means.to_dicts()}
            nadir = vals.get("0-15")
            if nadir is None:
                continue

            valid = [(VZA_LABELS.index(l), vals[l]) for l in VZA_LABELS if l in vals]
            for lbl in VZA_LABELS[1:]:  # skip nadir
                if lbl in vals:
                    row[f"{b}_vza_diff_{lbl.replace('-','_')}"] = vals[lbl] - nadir
            if "45-60" in vals and nadir != 0:
                row[f"{b}_vza_ratio_45"] = vals["45-60"] / nadir

            if len(valid) >= 2:
                x_vals = [v[0] for v in valid]
                y_vals = [v[1] for v in valid]
                n = len(x_vals)
                sx, sy = sum(x_vals), sum(y_vals)
                sxy = sum(x * y for x, y in zip(x_vals, y_vals))
                sxx = sum(x * x for x in x_vals)
                d = n * sxx - sx * sx
                slope = (n * sxy - sx * sy) / d if d != 0 else None
                row[f"{b}_angular_slope"] = slope

            existing = [vals[l] for l in VZA_LABELS if l in vals]
            if len(existing) >= 2:
                row[f"{b}_range"] = max(existing) - min(existing)
        rows_m5.append(row)
    m5 = pl.DataFrame(rows_m5)
    m5.write_parquet(OUT_DIR / "M5_angular_contrast.parquet")
    print(f"  M5: {m5.shape[0]} rows x {m5.shape[1]} cols")
    results["M5"] = m5

    print(f"\n=== Summary ({time.time() - t2:.1f}s for feature building) ===")
    print(f"  Total time: {time.time() - t0:.1f}s")
    for name, df in results.items():
        print(f"  {name}: {df.shape[0]} rows x {df.shape[1]} columns")


if __name__ == "__main__":
    main()
