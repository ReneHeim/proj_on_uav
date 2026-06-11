#!/usr/bin/env python3
"""Build aggregated analysis table for multiangular disease prediction.

Produces a clean table containing per-plot, per-week, per-band, per-vza-bin
reflectance statistics joined with polygon metadata (cultivar, treatment,
disease label).
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import numpy as np
import polars as pl
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths for all available weeks
# ---------------------------------------------------------------------------
WEEK_DEFS: Dict[str, Dict[str, str]] = {
    "2024_wk0": {
        "plot_dir": "/run/media/davidem/Heim/2024/20240603_week0/metashape/20241205_products_uav_data/output/extract/polygon_df",
        "year": 2024,
        "week": "wk0",
    },
    "2024_wk3": {
        "plot_dir": "/run/media/davidem/Heim/2024/20240624_week3/metashape/20241206_week3_products_uav_data/output/plots",
        "year": 2024,
        "week": "wk3",
    },
    "2024_wk5": {
        "plot_dir": "/run/media/davidem/Heim/2024/20240715_week5/metashape/20241207_week5_products_uav_data/output/plots",
        "year": 2024,
        "week": "wk5",
    },
    "2024_wk8": {
        "plot_dir": "/run/media/davidem/Heim/2024/20240826_week8/metashape/20241029_products_uav_data/output/extract/polygon_df",
        "year": 2024,
        "week": "wk8",
    },
    "2025_wk0": {
        "plot_dir": "/run/media/davidem/Heim/2025/week0/metashape/20250822_products_uas_data/output/plots",
        "year": 2025,
        "week": "wk0",
    },
    "2025_wk3": {
        "plot_dir": "/run/media/davidem/Heim/2025/week3/metashape/20250828_products_uas_data/output/plots",
        "year": 2025,
        "week": "wk3",
    },
    "2025_wk5": {
        "plot_dir": "/run/media/davidem/Heim/2025/week5/metashape/20250829_products_uas_data/output/plots",
        "year": 2025,
        "week": "wk5",
    },
}

POLYGON_PATH_2024 = "/run/media/davidem/Heim/2024_oncerco_plot_polygons.gpkg"
POLYGON_PATH_2025 = "/run/media/davidem/Heim/2025_oncerco_plot_polygons.gpkg"

# ---------------------------------------------------------------------------
# Band metadata
# ---------------------------------------------------------------------------
BAND_INFO: Dict[str, Dict[str, object]] = {
    "band1": {"name": "Blue", "wavelength_nm": 475},
    "band2": {"name": "Green", "wavelength_nm": 560},
    "band3": {"name": "Red", "wavelength_nm": 668},
    "band4": {"name": "Red Edge", "wavelength_nm": 717},
    "band5": {"name": "NIR", "wavelength_nm": 842},
}
BAND_COLS = list(BAND_INFO.keys())

# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------
VZA_BREAKS = [15, 25, 35, 45, 60]
VZA_LABELS = ["0-15", "15-25", "25-35", "35-45", "45-60", "60+"]

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
SAMPLE_SIZE = 500_000

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "tables"
PARQUET_PATH = OUT_DIR / "analysis_table_plot_week_band_angle.parquet"
CSV_PATH = OUT_DIR / "analysis_table_plot_week_band_angle.csv"


# ===================================================================
# Helpers
# ===================================================================


def load_polygon_metadata() -> Dict[int, Dict[int, dict]]:
    """Load polygon metadata for both years.

    Returns
    -------
    dict  year -> plot_idx -> {cult, ifz_id, trt, ino}
        plot_idx refers to the positional index in the gpkg, which maps
        to parquet filenames ``plot_{plot_idx}.parquet``.
    """
    meta: Dict[int, Dict[int, dict]] = {2024: {}, 2025: {}}

    # ---- 2024 ----
    gdf24 = gpd.read_file(POLYGON_PATH_2024)
    for idx, (_, row) in enumerate(gdf24.iterrows()):
        meta[2024][idx] = {
            "cult": row["cult"],
            "ifz_id": int(row["ifz_id"]),
            "trt": row["trt"],
            "ino": bool(row["ino"]) if row["ino"] is not None else None,
        }

    # ---- 2025 ----
    gdf25 = gpd.read_file(POLYGON_PATH_2025)
    for idx, (_, row) in enumerate(gdf25.iterrows()):
        meta[2025][idx] = {
            "cult": row["cultivar"],
            "ifz_id": 90001 + idx,
            "trt": row["trt"],
            "ino": None,
        }

    return meta


def discover_plot_files(plot_dir: str) -> List[int]:
    """Return sorted list of plot indices found in *plot_dir*."""
    plot_dir_p = Path(plot_dir)
    if not plot_dir_p.is_dir():
        return []
    indices = []
    for p in plot_dir_p.glob("plot_*.parquet"):
        stem = p.stem  # e.g. "plot_12"
        try:
            indices.append(int(stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(indices)


def process_plot_file(
    file_path: str,
    week_label: str,
    plot_idx: int,
    metadata: dict,
    sample_size: int = SAMPLE_SIZE,
) -> Optional[pl.DataFrame]:
    """Process a single per-plot parquet and return aggregated stats.

    The aggregation groups by ``(vza_bin, band)`` and computes
    reflectance statistics and angle summaries.

    Parameters
    ----------
    file_path : str
    week_label : str      e.g. "2024_wk0"
    plot_idx : int        positional index matching gpkg row
    metadata : dict       {cult, ifz_id, trt, ino} for this plot
    sample_size : int     max rows to read per file before melt

    Returns
    -------
    pl.DataFrame or None if there are no valid rows after filtering.
    """
    meta = metadata.get(plot_idx)
    if meta is None:
        tqdm.write(f"  [skip] plot_{plot_idx}: no metadata for index {plot_idx}")
        return None

    # ---- 1. Lazy scan ----
    lf = pl.scan_parquet(file_path)

    # ---- 2. Filter invalid pixels ----
    # Drop rows where any band is NaN or <= 0, or vza outside [0, 90]
    band_filters = [(pl.col(b).is_not_null() & (pl.col(b) > 0)) for b in BAND_COLS]
    lf = lf.filter(
        pl.all_horizontal(band_filters)
        & pl.col("vza").is_not_null()
        & pl.col("vza").is_between(0, 90)
    )

    # ---- 3. Count valid rows, sample if needed ----
    n_total = lf.select(pl.len()).collect().item()
    if n_total == 0:
        return None

    if n_total > sample_size:
        stride = max(1, n_total // sample_size)
        lf = lf.gather_every(stride)

    # ---- 4. Unpivot bands to long format ----
    lf_long = lf.unpivot(
        index=["plot_id", "vza", "vaa", "sunelev"],
        on=BAND_COLS,
        variable_name="band",
        value_name="reflectance",
    )

    # ---- 5. Add derived columns ----
    lf_long = lf_long.with_columns(
        pl.col("reflectance").cast(pl.Float64),
        ((pl.col("vza").cut(VZA_BREAKS, labels=VZA_LABELS))).alias("vza_bin"),
        (90.0 - pl.col("sunelev")).alias("sza"),
    )

    # ---- 6. Aggregate ----
    agg_exprs = [
        pl.col("reflectance").mean().alias("reflectance_mean"),
        pl.col("reflectance").median().alias("reflectance_median"),
        pl.col("reflectance").std().alias("reflectance_std"),
        pl.col("reflectance").quantile(0.10).alias("reflectance_p10"),
        pl.col("reflectance").quantile(0.90).alias("reflectance_p90"),
        pl.len().alias("n_pixels"),
        pl.col("vza").mean().alias("vza_mean"),
        pl.col("vaa").mean().alias("vaa_mean"),
    ]

    try:
        result = (
            lf_long.group_by(["plot_id", "vza_bin", "band"])
            .agg(agg_exprs)
            .collect(engine="streaming")
        )
    except Exception:
        tqdm.write(f"  [warn] plot_{plot_idx}: aggregation failed, trying without streaming")
        try:
            result = lf_long.group_by(["plot_id", "vza_bin", "band"]).agg(agg_exprs).collect()
        except Exception as e:
            tqdm.write(f"  [skip] plot_{plot_idx}: {e}")
            return None

    if result.is_empty():
        return None

    # ---- 7. Attach metadata columns ----
    result = result.with_columns(
        pl.lit(week_label).alias("week"),
        pl.lit(meta["cult"]).alias("cultivar"),
        pl.lit(meta["ifz_id"]).cast(pl.Int64).alias("ifz_id"),
        pl.lit(meta["trt"]).alias("treatment"),
    )

    # Disease label
    ino_val = meta.get("ino")
    result = result.with_columns(
        pl.lit(ino_val if ino_val is not None else None).alias("disease_label")
    )

    # ---- 8. Attach band name + wavelength ----
    band_name_map = {b: BAND_INFO[b]["name"] for b in BAND_COLS}
    band_wl_map = {b: BAND_INFO[b]["wavelength_nm"] for b in BAND_COLS}
    result = result.with_columns(
        pl.col("band").replace_strict(band_name_map).alias("band_name"),
        pl.col("band").replace_strict(band_wl_map).cast(pl.Int32).alias("wavelength_nm"),
    )

    # Drop rows with NULL vza_bin or "60+" (VZA > 60°)
    result = result.filter(pl.col("vza_bin").is_not_null() & (pl.col("vza_bin") != "60+"))

    return result


def build_table(
    sample_size: int = SAMPLE_SIZE,
    week_filter: Optional[List[str]] = None,
) -> pl.DataFrame:
    """Main entry point: build the full aggregated analysis table.

    Parameters
    ----------
    sample_size : int
        Max rows to read from each parquet file.
    week_filter : list of str or None
        If given, only process these week keys (e.g. ["2024_wk0"]).

    Returns
    -------
    pl.DataFrame
    """
    # ---- Metadata ----
    meta_all = load_polygon_metadata()

    # ---- Discover available weeks ----
    weeks_to_process = []
    for wkey, wdef in WEEK_DEFS.items():
        if week_filter and wkey not in week_filter:
            continue
        indices = discover_plot_files(wdef["plot_dir"])
        if indices:
            weeks_to_process.append((wkey, wdef, indices))
        else:
            tqdm.write(f"[skip] {wkey}: no parquet files found in {wdef['plot_dir']}")

    if not weeks_to_process:
        print("No data found.")
        return pl.DataFrame()

    # ---- Process weeks ----
    all_parts: List[pl.DataFrame] = []
    total_files = sum(len(indices) for _, _, indices in weeks_to_process)

    with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
        for wkey, wdef, indices in weeks_to_process:
            year = wdef["year"]
            plot_dir = wdef["plot_dir"]
            meta_year = meta_all[year]

            tqdm.write(f"\n--- {wkey} ({len(indices)} plots) ---")

            for pidx in indices:
                fpath = str(Path(plot_dir) / f"plot_{pidx}.parquet")
                if not Path(fpath).exists():
                    tqdm.write(f"  [skip] missing file: {fpath}")
                    pbar.update(1)
                    continue

                try:
                    df_part = process_plot_file(fpath, wkey, pidx, meta_year, sample_size)
                    if df_part is not None and not df_part.is_empty():
                        all_parts.append(df_part)
                except Exception as e:
                    tqdm.write(f"  [error] plot_{pidx}: {e}")

                pbar.update(1)

    if not all_parts:
        print("No data to combine.")
        return pl.DataFrame()

    # ---- Combine all parts ----
    print(f"\nCombining {len(all_parts)} dataframes...")
    combined = pl.concat(all_parts, how="vertical")

    # ---- Final column ordering ----
    final_columns = [
        "week",
        "plot_id",
        "ifz_id",
        "cultivar",
        "treatment",
        "disease_label",
        "band",
        "band_name",
        "wavelength_nm",
        "vza_bin",
        "n_pixels",
        "reflectance_mean",
        "reflectance_median",
        "reflectance_std",
        "reflectance_p10",
        "reflectance_p90",
        "vza_mean",
        "vaa_mean",
    ]

    # Ensure all columns exist
    for col in final_columns:
        if col not in combined.columns:
            combined = combined.with_columns(pl.lit(None).alias(col))

    combined = combined.select(final_columns)

    print(f"\nFinal table: {combined.height:,} rows, {len(combined.columns)} columns")
    print(f"Columns: {combined.columns}")

    return combined


# ===================================================================
# CLI
# ===================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build aggregated analysis table for multiangular disease prediction"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=SAMPLE_SIZE,
        help="Max rows to read per parquet file (default: %(default)s)",
    )
    parser.add_argument(
        "--week",
        type=str,
        nargs="*",
        default=None,
        help="Process only these week keys (e.g. 2024_wk0 2024_wk3)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUT_DIR),
        help="Output directory",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_out = out_dir / "analysis_table_plot_week_band_angle.parquet"
    csv_out = out_dir / "analysis_table_plot_week_band_angle.csv"

    # ---- Build ----
    df = build_table(sample_size=args.sample_size, week_filter=args.week)

    if df.is_empty():
        print("Empty result — nothing saved.")
        sys.exit(0)

    # ---- Save ----
    print(f"\nSaving to {parquet_out} ...")
    df.write_parquet(str(parquet_out))

    print(f"Saving to {csv_out} ...")
    df.write_csv(str(csv_out))

    print("Done.")
    print(f"\nRow count: {df.height:,}")
    print(f"Columns: {df.columns}")
