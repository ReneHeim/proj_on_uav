# Functions

### **DEM and Orthophoto Processing**

- **`read_dem`**: Reads DEM data, ensuring appropriate precision and data type conversion.
- **`retrieve_orthophoto_paths`**: Retrieves orthophoto file paths from configured directories.
- **`extract_sun_angles`**: Extracts solar elevation and azimuth angles from orthophoto metadata using EXIF data.
- **`get_camera_position`**: Reads and parses camera position data for each orthophoto from a CSV file.
- **`read_orthophoto_bands`**: Reads orthophoto raster bands and prepares the data for merging with the DEM.
- **`merge_data`**: Merges DEM and orthophoto data on aligned spatial coordinates for further analysis.
- **`calculate_angles`**: Calculates viewing zenith angle (VZA) and viewing azimuth angle (VAA) based on camera and DEM geometry.

### **Output Management**

- **`save_parquet`**: Saves processed data to a Parquet file with efficient compression.

---