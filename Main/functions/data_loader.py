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




def load_by_polygon(folder: str,
                    specific: Optional[str] = None) -> dict[str, pl.DataFrame]:
    """
    Loads all .parquet files from a folder and groups data by polygon (plot_id).

    Parameters:
    -----------
    folder : str
        Path to the folder containing parquet files.
    specific : str, optional
        If provided, only data for this polygon will be returned.

    Returns:
    --------
    A dictionary mapping plot_id to DataFrames containing only that polygon's data.
    """
    dfs = _read_folder(Path(folder))
    print(f"Loaded {len(dfs)} DataFrames from {folder}")

    polygons = {specific} if specific else unique_plot_ids(dfs)
    return split_by_polygon(dfs, polygons)



def create_polygon_dict(datarame_dict, polygon_list):
    """
    Creates a dictionary of polygons where each polygon has a dataframe with the data from that polygon.
    Params:
    -------
    - datarame_dict: dict
        Dictionary of dataframes with the data from all polygons.
    - polygon_list: list
        List of polygons to filter the dataframes.
    """
    polygon_dict = {}

    #Fill the dict with polygons and empty dataframes
    for polygon in polygon_list:
        polygon_dict[polygon] = pl.DataFrame()

    # Iterate through the dataframes and filter by polygon
    for file_name, df in datarame_dict.items():
        # Check if the DataFrame contains the 'plot_id' column
        if "plot_id" in df.columns:
            # Filter by polygon
            for polygon in polygon_list:
                filtered_df = df.filter(pl.col("plot_id") == polygon)
                if not filtered_df.is_empty():
                    # Append the filtered DataFrame to the corresponding polygon
                    polygon_dict[polygon] = pl.concat([polygon_dict[polygon], filtered_df])
        else:
            print(f"'plot_id' column not found in DataFrame {file_name}. Skipping filtering.")

        return polygon_dict



