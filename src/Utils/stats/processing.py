
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

from Common.preprocess import df_preprocess
from src.Utils.extract_data.raster import plotting_raster
from src.Utils.RPV_modelling.rpv import rpv_fit

def process_stats(dg, path, week, out):
    out.mkdir(parents=True, exist_ok=True)

    ## Plotting
    try:
        name = Path(path).stem

        # Convert sun elevation to zenith angle
        dg = dg.with_columns((90 - pl.col("sunelev")).alias("sza"))

        # Calculate relative azimuth angle constrained to [0,180]
        dg = dg.with_columns(
            (((pl.col("saa") - pl.col("vaa") + 180) % 360) - 180).alias("RAA")
        )

        #=== Plot distributions ===
        from src.Utils.stats.plotting import angle_kde_plot
        raa_edges = list(range(-360, 361, 90))
        vza_edges = [0, 20, 40, 60, 80]
        vza_bins = [(0, 20), (20, 40), (40, 60), (60, 80)]
        # Plot RAA distributions
        (out / "bands_distribution").mkdir(parents=True, exist_ok=True)
        for band in [f"band{i}" for i in range(1, 6)]:
            if (out / "bands_distribution" / f"{band}_{name}_vza.png").exists():
                logging.info(f"Skipping {name} {band} as it already exists")
                continue
            angle_kde_plot(dg, band=band, bins=vza_bins, points=1000, linewidth=1, colors=None, dpi=300,
                               xlim=None, angle='vza', out=out / "bands_distribution" / f"{band}_{name}_vza.png")

        #Do ANOVA
        from src.Utils.stats.ANOVA import ANOVA_optimized, ANOVA_preprocess
        dg = ANOVA_preprocess(dg, raa_edges=raa_edges, vza_edges=vza_edges)



        (out / "anova").mkdir(parents=True, exist_ok=True)
        if (out / "anova" / f"anova_results_{name}.csv").exists() == False:
            logging.info(f"Calculating ANOVA for {name}")
            anova_results = {}
            for band in [f"band{i}" for i in range(1, 6)]:
                anova_df = ANOVA_optimized(dg, band_col=band, col='raa_bin')
                anova_df = anova_df.with_columns(pl.lit(band).alias("band"))
                anova_results[band] = anova_df

            # Union all ANOVA results
            ANOVA_all = pl.concat(list(anova_results.values()))
            ANOVA_all.write_csv(out / 'anova' /f"anova_results_{name}.csv")
        else: logging.info(f"Skipping ANOVA for {name} as it already exists")


        # ==== Plot raster ====
        (out / "bands_data").mkdir(parents=True, exist_ok=True)
        if (out / "bands_data" / f"panels_{name}.png").exists():
            logging.info(f"Skipping {name} as it already exists")
        else:
            logging.info(f"Plotting raster for {name}")
            plotting_raster(
                dg,
                out,
                path.stem,
                custom_columuns=["OSAVI", "NDVI", "excess_green"],
                bands_prefix=None,
                debug=True,
                ny=380,
                nx=630,
                dpi=500,
                auto_figsize=True,
                density_discrete=True,
            )
    except Exception as e:
        raise e
        logging.error(f"Error in process_stats for {path}: {e}")
        return



def process_weekly_data_stats(weeks_dics, out, debug=False, filter={}):
    print(f"\n{'=' * 80}")
    print(f"{'Stats ANALYSIS STARTING':^80}")
    print(f"{'=' * 80}\n")

    # Create a list to collect all results
    all_results = []
    total_plots = sum(len(gdf) for gdf in weeks_dics.values())

    print(f"Total plots to process: {total_plots}\n")
    start_time = time.time()

    # Process each week
    for week, gdf in weeks_dics.items():
        print(f"\nProcessing {week.upper()} - {len(gdf)} plots")

        # Process each plot with a progress bar
        for row in tqdm(gdf.to_dicts(), desc=f"{week}", ncols=80):
            try:
                # Extract plot information
                plot_id = row.get("ifz_id", None)
                cult = row.get("cult", None)
                treatment = row.get("trt", None)
                geometry = row.get("geometry", None)

                dg = pl.read_parquet(row["paths"])

                dg = df_preprocess(dg, debug)

                if filter:
                    if filter["sign"] == ">":
                        dg = dg.filter(pl.col(filter["column"]) > filter["threshold"])
                    if filter["sign"] == "<":
                        dg = dg.filter(pl.col(filter["column"]) < filter["threshold"])

                process_stats(dg, path=Path(row["paths"]), week=week, out=out)

                # Add to results collection with proper types
                all_results.append(
                    {
                        "week": str(week) if week is not None else None,
                        "plot_id": (
                            int(plot_id)
                            if isinstance(plot_id, (int, str, float)) and plot_id is not None
                            else None
                        ),
                        "cultivar": str(cult) if cult is not None else None,
                        "treatment": str(treatment) if treatment is not None else None,
                        "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "status": "success",
                    }
                )
            except Exception as e:
                logging.warning(f"Error processing week {week}: {e}")
                all_results.append(
                    {
                        "week": str(week) if week is not None else None,
                        "plot_id": (
                            int(plot_id)
                            if isinstance(plot_id, (int, str, float)) and plot_id is not None
                            else None
                        ),
                        "cultivar": str(cult) if cult is not None else None,
                        "treatment": str(treatment) if treatment is not None else None,
                        "geometry": str(geometry) if geometry is not None else None,
                        "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "status": f"error: {str(e)[:100]}",
                    }
                )
