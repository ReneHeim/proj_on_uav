import logging

import polars as pl
def OSAVI_index_filtering(df):
    """
    Apply the OSAVI index filtering to the DataFrame.
    """
    try:
        # Ensure required columns exist
        required_columns = ["band5", "band3"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns for OSAVI calculation: {missing_columns}")

        # Calculate OSAVI index
        Y = 0.16
        df = df.with_columns([
            (1 + Y) * ((pl.col("band5") - pl.col("band3")) / (pl.col("band5") + pl.col("band3") + Y)).alias("OSAVI")
        ])

        # Verify OSAVI column creation
        if "OSAVI" not in df.columns:
            raise ValueError("Failed to create 'OSAVI' column in the DataFrame.")

        # Filter based on OSAVI index
        df = df.filter(pl.col("OSAVI") > 0.1)

        return df
    except Exception as e:
        logging.error(f"Error in OSAVI index filtering: {e}")
        raise