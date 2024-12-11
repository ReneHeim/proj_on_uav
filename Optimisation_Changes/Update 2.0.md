# Update 2.1

---
## Customization 

-  Added Precision Value in config_file.yaml



---
##  Refactoring 
- **Alpha**: Changed from Single Function to Multi function Currently in _Polars.py

---
## Optimisation 
- Dataframe managed from Pandas to Polars




# Update 2.1

---

## **Customization**

- **Precision Configuration**: Introduced a precision value in `config_file.yaml` to allow for customizable rounding of coordinates in data processing.

---

## **Refactoring**

- **Alpha Refactoring**: Refactored a single monolithic function into multiple modular functions for enhanced readability, maintainability, and scalability. Functions are now located in `_Polars.py`.

---

## **Optimization**

- **Dataframe Transition**: Replaced Pandas with Polars for improved performance and better handling of large datasets, especially in I/O and computational efficiency.

---

## **Enhancements**

### **Co-Registration Module**

- **`check_alignment`**: Verifies spatial alignment between the DEM and orthophoto datasets, including CRS consistency, pixel size, and grid alignment.
- **`coregister_and_resample`**: Reprojects and resamples orthophoto datasets to match the DEM’s spatial parameters, with optional support for pixel size adjustment.



## **Documentation**

- Added detailed comments and logging to each function for better traceability and debugging.
- Integrated validation steps to ensure data integrity, particularly for alignment and spatial consistency between datasets.

---

## **Testing and Validation**

- Verified alignment and resampling functionalities with test datasets to ensure accurate co-registration.
- Validated modularized processing pipeline for robustness and performance.

--- 

This update enhances the pipeline’s precision, modularity, and performance, ensuring it can handle large geospatial datasets efficiently while maintaining spatial integrity.