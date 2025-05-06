import polars as pl
import os
import glob

def create_polygon_dict(df, polygon):



def data_loader_from_file_lists(path,polygon = None):
    """
    This functions load all parquet files in the folder and create a dictionary of dataframes with the data from just one
    polygon, if None is provided, it will load all the dataframes in the folder.

    Params:
    -------
    - path: str
        Path to the folder containing the parquet files.
    - polygon: str
        Name of the polygon to filter the dataframes. If None, all dataframes will be loaded.
    """
    dataframes = {}

    try:
        # Get all parquet files in the folder
        paths = glob.glob(os.path.join(path, "*.parquet"))

        # Load each file into a DataFrame and store in the dictionary
        for file_path in paths:
            file_name = os.path.basename(file_path).split(".")[0]
            df = pl.read_parquet(file_path)

            # Split the dataframe in polygons dataframes
            polygon_list = df["plot_id"].unique().to_list()






            # Filter by polygon if provided
            if polygon is not None:
                if "plot_id"  in df.columns:
                    df = df.filter(pl.col("plot_id") == polygon)
                else:
                    print("'plot_id' column not found in DataFrame. Skipping filtering.")

            dataframes[file_name] = df





        return dataframes
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
