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
            ((1 + Y) * ((pl.col("band5") - pl.col("band3")) / (pl.col("band5") + pl.col("band3") + Y))).alias("OSAVI")
        ])

        # Verify OSAVI column creation

        # Filter based on OSAVI index
        len_before = len(df)
        df = df.filter(pl.col("OSAVI") > 0.7)

        logging.info(f"OSAVI filtering: {len_before} -> {len(df)} | Percentage of points filtered: {(len_before-len(df))/len_before * 100}%" )


        return df
    except Exception as e:
        logging.error(f"Error in OSAVI index filtering: {e}")
        raise
def excess_green_filter(df):
    """
    Apply the Excess Green index filtering to the DataFrame.
    """
    try:
        # Ensure required columns exist
        required_columns = ["band1", "band2", "band3"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns for Excess Green calculation: {missing_columns}")

        # Calculate Excess Green index
        df = df.with_columns([
            (2*pl.col("band2")-pl.col("band3")-pl.col("band1")).alias("ExcessGreen")
        ])

        len_before = len(df)

        # Filter based on Excess Green index
        df = df.filter(pl.col("ExcessGreen") > 0.05)
        logging.info(f"Excess Green filtering: {len_before} -> {len(df)} | Percentage of points filtered: {(len_before-len(df))/len_before * 100}%" )


        return df
    except Exception as e:
        logging.error(f"Error in Excess Green filtering: {e}")
        raise