import logging

import geopandas as gpd
import polars as pl
from matplotlib import pyplot as plt
from shapely.geometry.point import Point


def OSAVI_index_filtering(df, removal_threshold=None):
    """
    Compute the OSAVI index for each row and optionally filter out rows
    below a threshold.

    Args:
        df (polars.DataFrame): Input with 'band5' and 'band3' columns.
        removal_threshold (float, optional): If set, only keep rows with OSAVI > threshold.

    Returns:
        polars.DataFrame: DataFrame with new 'OSAVI' column, possibly filtered.

    Raises:
        ValueError if required columns are missing.

    Notes:
        - No in-place modification: always returns new DataFrame.
        - Will log and raise on error.
        - OSAVI: (1+0.16)*(NIR-Red)/(NIR+Red+0.16), with NIR=band5, Red=band3.
        - https://www.microimages.com/documentation/TechGuides/81VegIndices.pdf
    """
    try:
        # Ensure required columns exist
        required_columns = ["band5", "band3"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns for OSAVI calculation: {missing_columns}")

        # Calculate OSAVI index
        Y = 0.16
        df = df.with_columns(
            [
                (
                    (1 + Y)
                    * (
                        (pl.col("band5") - pl.col("band3"))
                        / (pl.col("band5") + pl.col("band3") + Y)
                    )
                ).alias("OSAVI")
            ]
        )

        # Filter based on OSAVI index
        if removal_threshold is not None:
            len_before = len(df)
            df = df.filter(pl.col("OSAVI") > removal_threshold)
            logging.info(
                f"OSAVI filtering: {len_before} -> {len(df)} | Percentage of points filtered: {(len_before - len(df)) / len_before * 100}%"
            )

        return df
    except Exception as e:
        logging.error(f"Error in OSAVI index filtering: {e}")
        raise


def excess_green_filter(df, removal_threshold=None):
    """
    Compute the Excess Green index and filter rows above a threshold.

    Args:
        df (polars.DataFrame): Input with 'band1', 'band2', 'band3'.
        removal_threshold (float, optional): If set, keeps only rows with ExcessGreen > threshold.

    Returns:
        polars.DataFrame: DataFrame with 'ExcessGreen' column, possibly filtered.

    Raises:
        ValueError if required columns are missing.

    Notes:
        - ExcessGreen = 2*Green - Red - Blue (band2, band3, band1).
        - Logging as above.
        - No in-place change.
    """
    try:
        # Ensure required columns exist
        required_columns = ["band1", "band2", "band3"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"Missing required columns for Excess Green calculation: {missing_columns}"
            )

        # Calculate Excess Green index
        df = df.with_columns(
            [(2 * pl.col("band2") - pl.col("band3") - pl.col("band1")).alias("ExcessGreen")]
        )

        if removal_threshold is not None:
            len_before = len(df)
            # Filter based on Excess Green index
            df = df.filter(pl.col("ExcessGreen") > removal_threshold)
            logging.info(
                f"Excess Green filtering: {len_before} -> {len(df)} | Percentage of points filtered: {(len_before-len(df))/len_before * 100}%"
            )

        return df
    except Exception as e:
        logging.error(f"Error in Excess Green filtering: {e}")
        raise


def plot_heatmap(df, column_name, output_path, sample_size=100000):
    """
    Create a 2D heatmap scatter plot of a numeric column over (Xw, Yw) coordinates.

    Args:
        df (polars.DataFrame or pandas.DataFrame): Input data, must include 'Xw', 'Yw', and column_name.
        column_name (str): The column to visualize.
        output_path (str): Directory to save plot.
        sample_size (int): Maximum points to plot (random sample).

    Returns:
        None. Saves PNG and shows the plot.

    Notes:
        - Will fail if columns missing.
        - Will overwrite previous PNG silently.
        - No file/folder existence check for output_path.
    """

    sample_df = df.sample(n=sample_size)

    # Convert to pandas if it's a polars dataframe
    if isinstance(sample_df, pl.DataFrame):
        sample_df = sample_df.to_pandas()

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create scatter plot with colormap
    scatter = ax.scatter(
        sample_df["Xw"],
        sample_df["Yw"],
        c=sample_df[column_name],
        cmap="viridis",
        s=10,  # marker size
        alpha=0.5,
    )

    # Add colorbar legend
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(column_name)

    # Add title and labels
    plt.title(f"Heatmap of {column_name}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")

    # Save the plot if output_path is provided
    if output_path:
        plt.savefig(f"{output_path}/heatmap_{column_name}.png", dpi=300)

    plt.show()


def plot_spectrogram(df, n_bands, bands_wavelength_list, sample_size=100000, output_path=None):
    """
    Plot average and std dev of reflectance vs. wavelength for all bands.

    Args:
        df (polars.DataFrame or pandas.DataFrame): Must contain band1 ... bandN.
        n_bands (int): Number of bands to plot.
        bands_wavelength_list (list of float): Wavelengths (nm) for each band.
        sample_size (int): Max number of samples.
        output_path (str, optional): If set, saves PNG.

    Returns:
        None. Shows and/or saves plot.

    Notes:
        - Requires n_bands == len(bands_wavelength_list).
        - Only mean ± 1 stddev region shown.
        - Will fail if required bands missing.
    """

    try:
        # Ensure we have the required columns
        band_columns = [f"band{i}" for i in range(1, n_bands + 1)]
        missing_columns = [col for col in band_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Check if wavelength list matches number of bands
        if len(bands_wavelength_list) != n_bands:
            raise ValueError(
                f"Number of wavelengths ({len(bands_wavelength_list)}) does not match number of bands ({n_bands})"
            )

        # Sample the dataframe if needed
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, replace=True)

        # Convert to pandas if it's a polars dataframe
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        # Calculate mean and std for each band
        means = [df[band].mean() for band in band_columns]
        stds = [df[band].std() for band in band_columns]

        # Create the plot
        plt.figure(figsize=(10, 6))

        # Plot mean reflectance
        plt.plot(bands_wavelength_list, means, "o-", color="blue", label="Mean Reflectance")

        # Plot error region (mean ± std)
        plt.fill_between(
            bands_wavelength_list,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            color="blue",
            alpha=0.2,
            label="±1 Std Dev",
        )

        # Add labels and title
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Reflectance")
        plt.title("Spectral Reflectance Profile")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()

        # Save the plot if requested
        if output_path:
            plt.savefig(f"{output_path}/spectrogram.png", dpi=300)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        logging.error(f"Error in plotting spectrogram: {e}")
        raise


def add_mask_and_plot(df, column_name, threshold, above=True, output_path=None):
    """
    Add a boolean mask column based on thresholding, and plot mask over (Xw, Yw).

    Args:
        df (polars.DataFrame or pandas.DataFrame): Must include 'Xw', 'Yw', column_name.
        column_name (str): Name of column to mask.
        threshold (float): Value to threshold at.
        above (bool): If True, mask = value > threshold. If False, mask = value < threshold.
        output_path (str, optional): If set, saves plot PNG.

    Returns:
        pandas.DataFrame: DataFrame with 'mask' column.

    Notes:
        - No check for output_path existence.
        - Always converts to pandas before plotting.
        - Mask column will be boolean.
        - Overwrites previous mask_plot PNG.
    """

    try:
        # Add mask column based on the 'above' parameter
        if above:
            df = df.with_columns((pl.col(column_name) > threshold).alias("mask"))
            comparison = ">"
        else:
            df = df.with_columns((pl.col(column_name) < threshold).alias("mask"))
            comparison = "<"

        # Convert to pandas if it's a polars dataframe
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        # Plot the result
        fig, ax = plt.subplots(figsize=(10, 10))
        scatter = ax.scatter(df["Xw"], df["Yw"], c=df["mask"], cmap="coolwarm", s=10, alpha=0.5)

        # Add colorbar legend
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Mask (True/False)")

        # Add title and labels
        plt.title(f"Mask Plot Based on {column_name} {comparison} {threshold}")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")

        # Save the plot if output_path is provided
        if output_path:
            plt.savefig(f"{output_path}/mask_plot_{column_name}.png", dpi=300)

        plt.show()

        return df
    except Exception as e:
        logging.error(f"Error in adding mask and plotting: {e}")
        raise
