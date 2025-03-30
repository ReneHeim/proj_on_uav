# UAV Orthophoto and DEM Processing Documentation

This document provides a detailed explanation of the processing pipeline for UAV-acquired orthophotos and digital elevation models (DEMs). The pipeline extracts pixel-level data, performs coordinate transformations, and calculates viewing geometry.

## Table of Contents

1. [Overview and Workflow](#overview-and-workflow)
2. [Configuration and Setup](#configuration-and-setup)
3. [Data Reading Functions](#data-reading-functions)
4. [Data Processing Functions](#data-processing-functions)
5. [Alignment and Registration](#alignment-and-registration)
6. [Output Handling](#output-handling)
7. [Utility Functions](#utility-functions)
8. [Debugging and Analysis](#debugging-and-analysis)

---

## Overview and Workflow

The processing pipeline follows this general workflow:

```
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│  Read DEM Data    │     │ Read Orthophoto   │     │ Get Camera        │
│  with Coordinates │     │ Bands & Coords    │     │ Position          │
└─────────┬─────────┘     └─────────┬─────────┘     └─────────┬─────────┘
          │                         │                         │
          └─────────────┬───────────┘                         │
                        ▼                                     │
           ┌───────────────────────────┐                      │
           │ Merge Data Based on       │                      │
           │ Coordinate Pairs (X,Y)    │                      │
           └─────────────┬─────────────┘                      │
                         │                                    │
                         ▼                                    ▼
             ┌─────────────────────┐              ┌────────────────────┐
             │ Calculate Solar     │◄─────────────┤ Extract Sun Angles │
             │ Position            │              └────────────────────┘
             └─────────┬───────────┘
                       │
                       ▼
             ┌─────────────────────┐
             │ Calculate Viewing   │
             │ Angles & Geometry   │
             └─────────┬───────────┘
                       │
                       ▼
             ┌─────────────────────┐
             │ Save to Parquet     │
             │ Format              │
             └─────────────────────┘
```

---

## Configuration and Setup

### `config_objecture_logging()`

Sets up the logging configuration for the application with both file and console outputs.

```python
def config_objecture_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("process.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
```

**Purpose**: Establishes consistent logging across the application with timestamps, log levels, and multiple outputs.

---

## Data Reading Functions

### `read_dem(dem_path, precision, transform_to_utm=True, target_crs="EPSG:32632")`

Reads a Digital Elevation Model (DEM) raster and transforms coordinates to the target coordinate reference system.

**Parameters**:
- `dem_path`: Path to the DEM file
- `precision`: Decimal precision for coordinate rounding
- `transform_to_utm`: Whether to transform coordinates to UTM
- `target_crs`: Target coordinate reference system

**Returns**: Polars DataFrame with columns `Xw`, `Yw`, and `elev`

**Process**:
1. Opens the DEM raster file
2. Extracts elevation data as a numpy array
3. Creates a grid of row and column indices
4. Converts indices to real-world coordinates using the raster transform
5. Optionally transforms coordinates to UTM or other CRS
6. Creates a DataFrame with coordinates and elevation values
7. Rounds coordinates to specified precision and removes duplicates

```
┌─────────┐     ┌─────────────────┐     ┌───────────────────┐
│ DEM     │     │ Convert Pixel   │     │ Transform to      │
│ Raster  │────►│ Indices to      │────►│ Target CRS        │
│         │     │ Coordinates     │     │ (e.g., UTM)       │
└─────────┘     └─────────────────┘     └─────────┬─────────┘
                                                  │
                                                  ▼
                                        ┌───────────────────┐
                                        │ Create DataFrame  │
                                        │ with Xw, Yw, elev │
                                        └─────────┬─────────┘
                                                  │
                                                  ▼
                                        ┌───────────────────┐
                                        │ Round Coordinates │
                                        │ & Remove Duplicate│
                                        └───────────────────┘
```

### `read_orthophoto_bands(each_ortho, precision, transform_to_utm=True, target_crs="EPSG:32632")`

Reads an orthophoto raster and extracts band values with corresponding coordinates.

**Parameters**:
- `each_ortho`: Path to the orthophoto file
- `precision`: Decimal precision for coordinate rounding
- `transform_to_utm`: Whether to transform coordinates to UTM
- `target_crs`: Target coordinate reference system

**Returns**: Polars DataFrame with columns `Xw`, `Yw`, and band value columns (`band1`, `band2`, etc.)

**Process**:
1. Opens the orthophoto raster file
2. Reads all bands as a numpy array
3. Creates a grid of row and column indices
4. Converts indices to real-world coordinates using the raster transform
5. Optionally transforms coordinates to UTM or other CRS
6. Creates a DataFrame with coordinates and band values
7. Rounds coordinates to specified precision and removes duplicates

---

## Data Processing Functions

### `merge_data(df_dem, df_allbands, precision, debug="verbose")`

Merges DEM and orthophoto data based on matching coordinate pairs.

**Parameters**:
- `df_dem`: Polars DataFrame with DEM data
- `df_allbands`: Polars DataFrame with orthophoto band data
- `precision`: Decimal precision for coordinate rounding
- `debug`: Debug level ("verbose" enables detailed analytics)

**Returns**: Polars DataFrame with merged DEM elevation and orthophoto band values

**Process**:
1. Groups DEM data by coordinates and calculates mean elevation for each unique coordinate pair
2. Rounds coordinates in both datasets to specified precision
3. Joins the DEM and orthophoto data on coordinates using an inner join
4. If debug is enabled, calculates and logs detailed statistics about the merge operation

**Merge Data Visualization**:
```
DEM DataFrame              Orthophoto DataFrame
┌────┬────┬─────┐          ┌────┬────┬────────┬────────┬────────┐
│ Xw │ Yw │elev │          │ Xw │ Yw │ band1  │ band2  │ band3  │
├────┼────┼─────┤          ├────┼────┼────────┼────────┼────────┤
│563.4│571.3│ 125.7│       │563.4│571.3│ 0.325│ 0.441 │  0.112 │
│564.2│571.4│ 126.1│       │564.2│571.4│ 0.331│ 0.445 │  0.115 │
│563.9│571.5│ 124.9│       │563.9│571.5│ 0.328│ 0.442 │  0.110 │
└────┴────┴─────┘          └────┴────┴────────┴────────┴────────┘
           │                              │
           └──────────────┬───────────────┘
                          ▼
        Merged DataFrame (Inner Join on Xw, Yw)
┌────┬────┬─────┬────────┬────────┬────────┐
│ Xw │ Yw │elev │ band1  │ band2  │ band3  │
├────┼────┼─────┼────────┼────────┼────────┤
│563.4│571.3│ 125.7│  0.325 │  0.441 │  0.112 │
│564.2│571.4│ 126.1│  0.331 │  0.445 │  0.115 │
│563.9│571.5│ 124.9│  0.328 │  0.442 │  0.110 │
└────┴────┴─────┴────────┴────────┴────────┘
```

### `calculate_angles(df_merged, xcam, ycam, zcam, sunelev, saa)`

Calculates viewing geometry angles for each pixel based on camera position and solar position.

**Parameters**:
- `df_merged`: Merged DataFrame with coordinates and elevation
- `xcam`, `ycam`, `zcam`: Camera position coordinates
- `sunelev`: Sun elevation angle in degrees
- `saa`: Sun azimuth angle in degrees

**Returns**: DataFrame with additional columns for viewing angles and geometry

**Process**:
1. Calculates delta_z, delta_x, delta_y as differences from camera position
2. Calculates planar (xy) distance from camera
3. Calculates view zenith angle (VZA) using arctangent of delta_z/distance_xy
4. Calculates view azimuth angle (VAA) using arctangent of delta_x/delta_y
5. Normalizes VAA relative to the sun azimuth angle
6. Filters out outlier elevation values (outside the central 90%)
7. Adds camera position and sun angle metadata

**Viewing Geometry Diagram**:
```
                      Camera
                        │
                        ▼
                        O
                       /│\
                      / │ \
                     /  │  \
                    /   │   \
                   /   VZA   \
                  /     │     \
                 /      │      \
       View ray /       │       \
               /        │        \
              /         │         \
     Pixel 1 □          │          □ Pixel 2
              \         │         /
                 Ground Surface
```

---

## Alignment and Registration

### `check_alignment(dem_path, ortho_path)`

Checks if a DEM and orthophoto are properly aligned in terms of CRS, pixel size, and pixel boundaries.

**Parameters**:
- `dem_path`: Path to the DEM file
- `ortho_path`: Path to the orthophoto file

**Returns**: Boolean indicating whether the files are aligned

**Process**:
1. Opens both raster files
2. Compares coordinate reference systems
3. Compares pixel resolutions
4. Checks if pixel grids are aligned (origins offset by integer multiples of pixel size)

### `coregister_and_resample(input_path, ref_path, output_path, target_resolution=None, resampling=Resampling.nearest)`

Reprojects and resamples an input raster to match a reference raster.

**Parameters**:
- `input_path`: Path to the input raster (e.g., orthophoto)
- `ref_path`: Path to the reference raster (e.g., DEM)
- `output_path`: Path for the output coregistered raster
- `target_resolution`: Optional tuple (xres, yres) for output resolution
- `resampling`: Resampling algorithm to use

**Returns**: Path to the output coregistered raster

**Process**:
1. Reads reference raster metadata (CRS, transform, dimensions, bounds)
2. Reads input raster
3. Calculates transform for output raster
4. Creates output raster with matching properties
5. Reprojects each band of the input to match the reference

---

## Output Handling

### `save_parquet(df, out, source, iteration, file)`

Saves a DataFrame to a compressed Parquet file.

**Parameters**:
- `df`: DataFrame to save
- `out`: Output directory
- `source`: Source metadata dictionary
- `iteration`: Current iteration number
- `file`: Current file name

**Process**:
1. Constructs output filename
2. Saves DataFrame to Parquet format with zstd compression

---

## Utility Functions

### `retrieve_orthophoto_paths(ori)`

Retrieves all TIFF files from specified directories.

**Parameters**:
- `ori`: List of directories to search

**Returns**: List of orthophoto file paths

### `fix_path(path_str)`

Normalizes file path separators.

**Parameters**:
- `path_str`: Path string to normalize

**Returns**: Normalized path string

### `extract_sun_angles(name, lon, lat, datetime_str, timezone="UTC")`

Calculates sun position (elevation and azimuth) for a given location and time.

**Parameters**:
- `name`: Image name for logging
- `lon`, `lat`: Longitude and latitude coordinates
- `datetime_str`: Date and time string (format: 'YYYY-MM-DD HH:MM:SS')
- `timezone`: Timezone string

**Returns**: Tuple of (sun_elevation, sun_azimuth) in degrees

**Process**:
1. Parses the datetime string
2. Converts to the specified timezone
3. Uses the pysolar library to calculate sun position
4. Normalizes azimuth to the range [0, 360]

### `get_camera_position(cam_path, name)`

Retrieves camera position from a camera position file.

**Parameters**:
- `cam_path`: Path to the camera position file
- `name`: Image name to look up

**Returns**: Tuple of (lon, lat, zcam) coordinates

**Process**:
1. Reads camera position file as a CSV
2. Filters for the specified image name
3. Extracts X, Y, Z coordinates

### `filter_df_by_polygon(df, polygon_path, target_crs="EPSG:32632", precision=2)`

Filters DataFrame points to include only those within specified polygon boundaries.

**Parameters**:
- `df`: DataFrame with Xw, Yw coordinate columns
- `polygon_path`: Path to polygon file (.shp)
- `target_crs`: Target coordinate reference system
- `precision`: Coordinate precision

**Returns**: Filtered DataFrame

**Process**:
1. Reads polygon shapefile using geopandas
2. Ensures CRS is correct
3. Creates a union of all valid polygons
4. Converts DataFrame coordinates to points
5. Filters to include only points within the polygon

---

## Core Processing Functions

### `process_orthophoto(each_ortho, cam_path, path_flat, out, source, iteration, exiftool_path, precision, polygon_filtering=False, alignment=False)`

Main function for processing a single orthophoto image.

**Parameters**:
- `each_ortho`: Path to the orthophoto file
- `cam_path`: Path to camera position file
- `path_flat`: List of all orthophoto paths
- `out`: Output directory
- `source`: Source metadata dictionary
- `iteration`: Current iteration number
- `exiftool_path`: Path to exiftool executable
- `precision`: Coordinate precision
- `polygon_filtering`: Whether to filter by polygon
- `alignment`: Whether to check and enforce alignment

**Process**:
1. Gets camera position
2. Optionally checks if the image is within a polygon
3. Optionally checks and fixes alignment issues
4. Reads DEM data
5. Reads orthophoto band data
6. Merges DEM and orthophoto data
7. Optionally filters by polygon
8. Calculates sun angles
9. Calculates viewing angles
10. Adds filename to the data
11. Saves processed data to Parquet format

### `build_database(tuple_chunk, source, exiftool_path)`

Processes a chunk of orthophoto images.

**Parameters**:
- `tuple_chunk`: Tuple of (iteration, image_list)
- `source`: Source metadata dictionary
- `exiftool_path`: Path to exiftool executable

**Process**:
1. Extracts iteration and image list
2. Sets up processing parameters
3. Retrieves orthophoto paths
4. Processes each orthophoto in the chunk

### `main()`

Main function that initializes processing for all images.

**Process**:
1. Sets up logging
2. Loads configuration from YAML file
3. Constructs source metadata
4. Processes each image sequentially

---

## Debugging and Analysis

### `visualize_coordinate_alignment(df_dem, df_allbands, precision)`

Visualizes how well coordinates align between DEM and orthophoto datasets.

**Parameters**:
- `df_dem`: DEM DataFrame
- `df_allbands`: Orthophoto bands DataFrame
- `precision`: Coordinate precision

**Returns**: Dictionary with overlap statistics

**Process**:
1. Rounds coordinates to specified precision
2. Compares coordinate sets between datasets
3. Calculates and visualizes overlap percentages
4. Generates scatter plot showing coordinate distribution

### `analyze_kdtree_matching(df_dem, df_allbands, precision, max_distance=1.0)`

Analyzes potential matches between datasets using K-d tree spatial indexing.

**Parameters**:
- `df_dem`: DEM DataFrame
- `df_allbands`: Orthophoto bands DataFrame
- `precision`: Coordinate precision
- `max_distance`: Maximum distance threshold for "near" matches

**Returns**: Dictionary with matching statistics

**Process**:
1. Extracts coordinates as numpy arrays
2. Builds K-d tree for band coordinates
3. Queries nearest neighbors for DEM points
4. Counts exact matches, near matches, and non-matches
5. Calculates and logs percentages and performance metrics

---

## Example Processing Workflow

For a complete UAV dataset processing:

1. **Preparation**:
   - Configure settings in `config_file.yaml`
   - Ensure DEM and orthophoto files are accessible

2. **Execution**:
   - Run the main script
   - Monitor progress through logging output

3. **Output**:
   - Parquet files containing merged data with viewing geometry
   - Analysis visualizations (if debug is enabled)

4. **Performance Considerations**:
   - DEM reading is typically the most time-consuming operation
   - Coordinate transformations can be slow for large datasets
   - The merge operation is highly optimized using lazy evaluation

---

## Data Flow Diagram

```
┌─────────────────┐
│ Configuration   │
│ YAML File       │
└───────┬─────────┘
        │
        ▼
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│  DEM File         │     │  Orthophoto       │     │  Camera Position  │
│  (.tif)           │     │  Files (.tif)     │     │  File (.txt)      │
└─────────┬─────────┘     └─────────┬─────────┘     └─────────┬─────────┘
          │                         │                         │
          ▼                         ▼                         ▼
┌─────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│ Read and        │     │ Read and          │     │ Extract Camera    │
│ Transform DEM   │     │ Transform Bands   │     │ Positions         │
└─────────┬───────┘     └─────────┬─────────┘     └─────────┬─────────┘
          │                       │                         │
          └───────────┬───────────┘                         │
                      ▼                                     │
         ┌─────────────────────────┐                        │
         │ Merge Data              │                        │
         └─────────────┬───────────┘                        │
                       │                                    │
                       ▼                                    ▼
         ┌─────────────────────────┐            ┌────────────────────┐
         │ Calculate Viewing       │◄───────────┤ Calculate Sun      │
         │ Geometry                │            │ Position            │
         └─────────────┬───────────┘            └────────────────────┘
                       │
                       ▼
         ┌─────────────────────────┐
         │ Save to Parquet         │
         └─────────────────────────┘
