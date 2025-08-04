import logging
from itertools import islice
from pathlib import Path
from typing import Optional
import polars as pl


def _read_folder(folder: Path) -> list[pl.DataFrame]:
    """
    Reads all Parquet files from a given folder into memory.

    Parameters:
    -----------
    folder : Path
        The folder containing .parquet files.

    Returns:
    --------
    List of Polars DataFrames, one per file.
    """
    return [pl.read_parquet(p) for p in folder.glob("*.parquet")]


def unique_plot_ids(dataframes: list[pl.DataFrame]) -> set[str]:
    """
    Collects all unique 'plot_id' values across multiple DataFrames.

    Parameters:
    -----------
    dataframes : list of pl.DataFrame
        The dataframes to search for 'plot_id' values.

    Returns:
    --------
    A set of all unique plot_id values.
    """
    ids: set[str] = set()
    for df in dataframes:
        if "plot_id" in df.columns:
            ids.update(df["plot_id"].unique().to_list())
    return ids


def split_by_polygon(dataframes: list[pl.DataFrame],
                     polygons: set[str]) -> dict[str, pl.DataFrame]:
    """
    Filters and groups rows by plot_id (polygon) across all input DataFrames.

    Parameters:
    -----------
    dataframes : list of pl.DataFrame
        Input dataframes possibly containing 'plot_id' columns.
    polygons : set of str
        The plot_id values to extract.

    Returns:
    --------
    Dictionary mapping each plot_id to its corresponding concatenated DataFrame.
    """
    return {
        p: pl.concat([
            df.filter(pl.col("plot_id") == p)
            for df in dataframes if "plot_id" in df.columns
        ])
        for p in polygons
    }


def stream_to_parquet(src: Path, polygons: set[str], out_dir: Path) -> None:
    """
    Streams rows for each specified polygon from all Parquet files in a source folder and writes them to separate Parquet files in an output directory.

    This function processes data in a memory-efficient way using Polars' lazy API, ensuring that only the relevant data for each polygon is written to disk.
    Each polygon's data is saved as a separate file named '{plot_id}.parquet' in the output directory.

    Parameters:
    -----------
    src : Path
        Path to the folder containing source .parquet files.
    polygons : set[str]
        Set of 'plot_id' values (polygons) to extract and write.
    out_dir : Path
        Path to the output directory where the per-polygon Parquet files will be saved.

    Returns:
    --------
    None
    """
    out_dir = out_dir.resolve()           # guarantees absolute path
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Writing to {str(out_dir)}" )          # tells you once, up front

    for p in polygons:
        target = out_dir / f"{p}.parquet"
        (pl.scan_parquet(src / "*.parquet")
             .filter(pl.col("plot_id") == p)
             .sink_parquet(target))
        logging.info(f"  • wrote {str(target)}")


def files_without_plot_id(folder: Path) -> list[Path]:
    """
    Scans all Parquet files in a folder and returns the list of files missing the 'plot_id' column.

    This function uses Polars' schema reader to check file metadata only, without loading full data.

    Parameters:
    -----------
    folder : Path
        Path to the folder containing .parquet files to scan.

    Returns:
    --------
    list[Path]
        A list of Parquet file paths that do not contain the 'plot_id' column.
    """
    missing = []
    for fp in folder.glob("*.parquet"):
        try:
            if "plot_id" not in pl.read_parquet_schema(fp):
                missing.append(fp)
        except Exception as err:
            print(f"{fp.name}: {err}")  # e.g. corrupted footer
    return missing


def batched(it, n):
    """
    Yields batches of size `n` from an input iterable.

    Useful for splitting long file lists into chunks to avoid resource exhaustion (e.g., open file limits).

    Parameters:
    -----------
    it : Iterable
        An iterable to split into batches.
    n : int
        Maximum number of elements per batch.

    Yields:
    -------
    list
        A batch (list) of up to `n` elements from the original iterable.
    """
    it = iter(it)
    while (batch := list(islice(it, n))):
        yield batch



def unique_plot_ids_scan(folder: Path, batch_size=100) -> set[str]:
    """
    Scans all Parquet files in a folder and returns the set of unique 'plot_id' values found.

    This function uses Polars' lazy API to efficiently collect unique plot identifiers without reading all data into memory.

    Parameters:
    -----------
    folder : Path
        Path to the folder containing .parquet files to scan.

    Returns:
    --------
    set[str]
        A set of all unique 'plot_id' values found in the folder's Parquet files.
    """

    #search for files without plotid:

    bad = files_without_plot_id(Path(r"D:\20240603_week0\metashape\20241205_products_uav_data\output\extract"))
    if len(bad) > 0:
        print(f"Found band {bad} files")
        print(f"{len(bad)} files lack plot_id")

    uids = set()
    #Batches



    for files in batched(folder.glob("*.parquet"), batch_size):
        ids = (pl.scan_parquet(files)
                 .select("plot_id")
                 .unique()
                 .collect())["plot_id"]
        uids.update(ids.to_list())
    return uids

def load_by_polygon(df_folder: str,
                    df_polygon_folder: str,
                    specific: Optional[str] = None) -> str:
    """
    Loads all .parquet files from a folder and writes each polygon's data to separate files,
    minimizing RAM usage by not returning dataframes.

    Parameters:
    -----------
    df_folder : str
        Path to the folder containing parquet files.
    df_polygon_folder : str
        Folder where the per-polygon parquet files will be saved.
    specific : str, optional
        If provided, only data for this polygon will be processed.

    Returns:
    --------
    "done" when operation is complete.
    """
    polygons = {specific} if specific else unique_plot_ids_scan(Path(df_folder))
    stream_to_parquet(Path(df_folder), polygons, Path(df_polygon_folder))
    return "done"


def create_polygon_dict(dataframe_dict: dict, polygon_list: list) -> dict:
    """
    Creates a dictionary mapping each polygon to a DataFrame containing all rows for that polygon
    across all input DataFrames.

    Parameters:
    -----------
    dataframe_dict : dict
        Dictionary of DataFrames (key: file name, value: pl.DataFrame).
    polygon_list : list
        List of polygon IDs (plot_id values) to extract.

    Returns:
    --------
    dict
        Dictionary mapping each polygon ID to a concatenated DataFrame of its data.
    """
    polygon_dict = {polygon: pl.DataFrame() for polygon in polygon_list}

    for file_name, df in dataframe_dict.items():
        if "plot_id" in df.columns:
            for polygon in polygon_list:
                filtered_df = df.filter(pl.col("plot_id") == polygon)
                if not filtered_df.is_empty():
                    # Combine DataFrames for each polygon
                    polygon_dict[polygon] = pl.concat([polygon_dict[polygon], filtered_df])
        else:
            logging.warning(f"'plot_id' column not found in DataFrame {file_name}. Skipping filtering.")

    return polygon_dict