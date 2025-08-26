import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

from src.Utils.extract_data.raster import plotting_raster
from src.Utils.RPV_modelling.rpv import rpv_fit


def df_preprocess(df, debug=False, load_indeces=False):
    EPS = 1e-2  # Tolerance for floating point comparison

    #Drop NaN and nulls
    len_before = len(df)
    df = df.fill_nan(None)
    df = df.drop_nulls()
    logging.info(f"Dropped {len_before - len(df)} NaN and nulls the: {round((len_before - len(df))/len_before * 100,3)} %")



    if "vx" in df.columns:
        vx = df["vx"]
        assert np.allclose(vx, df["xcam"] - df["Xw"], atol=EPS), "vx mismatch"
    else:
        vx = df["xcam"] - df["Xw"]
        df = df.with_columns(pl.Series("vx", vx))

    if "vy" in df.columns:
        vy = df["vy"]
        assert np.allclose(vy, df["ycam"] - df["Yw"], atol=EPS), "vy mismatch"
    else:
        vy = df["ycam"] - df["Yw"]
        df = df.with_columns(pl.Series("vy", vy))

    if "vz" in df.columns:
        vz = df["vz"]
        assert np.allclose(vz, df["delta_z"], atol=EPS), "vz mismatch"
    else:
        vz = df["delta_z"]
        df = df.with_columns(pl.Series("vz", vz))

    if "v_norm" in df.columns:
        v_norm = df["v_norm"]
        assert np.allclose(v_norm, np.sqrt(vx ** 2 + vy ** 2 + vz ** 2), atol=EPS), "v_norm mismatch"
    else:
        v_norm = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
        df = df.with_columns(pl.Series("v_norm", v_norm))

    # Formulas as expressions
    cos_vza = (df["vz"] / (df["v_norm"] + 1e-12)).clip(-1.0, 1.0)  # 1  guard /0 and range
    vza_formula = np.degrees(np.arccos(cos_vza))
    vaa_formula = np.where(
        (df["vx"].to_numpy() == 0) & (df["vy"].to_numpy() == 0),
        np.nan,
        np.degrees(np.arctan2(df["vx"].to_numpy(), df["vy"].to_numpy())) % 360
    )

    sza_formula = 90 - df["sunelev"]

    raa_formula = np.abs(df["saa"] - vaa_formula) % 360
    raa_formula = np.where(raa_formula <= 180, raa_formula, 360 - raa_formula)

    # indexes
    ndvi_formula = (df["band5"] - df["band3"]) / (df["band5"] + df["band3"])
    excess_green_formula = (2 * df["band2"] - df["band3"] - df["band1"])

    Y = 0.16
    osavi_formula = (1 + Y) * (df["band5"] - df["band3"]) / (df["band5"] + df["band3"] + Y)
    #if load_indeces:
        #TODO load indeces

    # Check or create columns
    for col, formula in [
        ("vza", vza_formula),
        ("vaa", vaa_formula),
        ("sza", sza_formula),
        ("NDVI", ndvi_formula),
        ("raa", raa_formula),
        ("excess_green", excess_green_formula),
        ("OSAVI", osavi_formula),
    ]:
        if col in df.columns and debug == True:
            if not np.allclose(df[col], formula, atol=EPS):
                print(f"Column '{col}' values do not match formula!")
                df = df.with_columns(pl.Series(col, formula))
        else:
            df = df.with_columns(pl.Series(col, formula))
    return df


def process_weekly_data_rpv(weeks_dics, band, debug=False, n_samples_bins=5000, sample_total_dataset=None , filter ={}):
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
                        dg = dg.sample(sample_total_dataset,with_replacement=True)
                    else:
                        dg = dg.sample(sample_total_dataset)
                dg = df_preprocess(dg, debug)


                if filter:
                    if filter["sign"] == ">":
                        dg = dg.filter(pl.col(filter["column"]) > filter["threshold"] )
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

def process_stats(dg, path, week, out):
        try:
            name = Path(path).stem
            if (out / "bands_data" / f"panels_{name}.png").exists():
                logging.info(f"Skipping {name} as it already exists")
                return
        except Exception as e:
            print(e)
            return


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








def process_weekly_data_stats(weeks_dics, out ,debug=False, filter = {}):
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
                        dg = dg.filter(pl.col(filter["column"]) > filter["threshold"] )
                    if filter["sign"] == "<":
                        dg = dg.filter(pl.col(filter["column"]) < filter["threshold"])


                process_stats(dg,path=Path(row["paths"]),week = week , out = out)

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


