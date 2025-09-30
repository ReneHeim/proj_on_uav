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


def process_weekly_data_rpv(
    weeks_dics, band, debug=False, n_samples_bins=5000, sample_total_dataset=None, filter={}
):
    """
    Process RPV data for each week and return results as a Polars DataFrame

    Args:
        weeks_dics (dict): Dictionary with week IDs as keys and dataframes as values
        filter (dict): Filter a column with a treshholdthe data before fitting the model.

    Returns:
        pl.DataFrame: A Polars DataFrame containing all RPV analysis results
    """
    print(f"\n{'=' * 80}")
    print(f"{'RPV ANALYSIS STARTING':^80}")
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

                if sample_total_dataset is not None:
                    if len(dg) < sample_total_dataset:
                        dg = dg.sample(sample_total_dataset, with_replacement=True)
                    else:
                        dg = dg.sample(sample_total_dataset)
                dg = df_preprocess(dg, debug)

                if filter:
                    if filter["sign"] == ">":
                        dg = dg.filter(pl.col(filter["column"]) > filter["threshold"])
                    if filter["sign"] == "<":
                        dg = dg.filter(pl.col(filter["column"]) < filter["threshold"])

                rpv_result = rpv_fit(dg, n_samples_bins=n_samples_bins, band=band)
                rho0, k, theta, rc, rmse, nrmse = rpv_result

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
                        "rho0": float(rho0) if rho0 is not None else None,
                        "k": float(k) if k is not None else None,
                        "theta": float(theta) if theta is not None else None,
                        "rc": float(rc) if rc is not None else None,
                        "rmse": float(rmse) if rmse is not None else None,
                        "nrmse": float(nrmse) if nrmse is not None else None,
                        "geometry": str(geometry) if geometry is not None else None,
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
                        "rho0": None,
                        "k": None,
                        "theta": None,
                        "rc": None,
                        "rmse": None,
                        "nrmse": None,
                        "geometry": str(geometry) if geometry is not None else None,
                        "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "status": f"error: {str(e)[:100]}",
                    }
                )

    # Create the final Polars DataFrame
    results_df = pl.DataFrame(all_results)

    # Print summary
    elapsed_time = time.time() - start_time
    success_count = results_df.filter(pl.col("status") == "success").height
    error_count = total_plots - success_count

    print(f"\n{'=' * 80}")
    print(f"{'ANALYSIS COMPLETE':^80}")
    print(f"Processed {total_plots} plots in {elapsed_time:.2f} seconds")
    print(f"Success: {success_count} | Errors: {error_count}")
    print(f"{'=' * 80}\n")

    # Show first few rows
    if not results_df.is_empty():
        print("\nSample of results:")
        print(results_df.head(5))

    return results_df
