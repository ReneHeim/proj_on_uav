# Multi-angular UAV Reflectance Extractor

## Why is this repository important to you?

This repository contains code, written in [Python](https://www.python.org/), to reproduce [LINK TO ARTCLE HERE]. If you are using any of the contained code or data, please use the following reference:

Heim, R. HJ., Okole, N., Steppe, K., van Labeke, M.C., Geedicke, I., & Maes, W. H. (2024). An applied framework to unlocking multi‑angular UAV reflectance data: A case study for classification of plant parameters in maize (*Zea mays*). *Precision Agriculture*. (accepted)

![alt text](https://github.com/ReneHeim/proj_on_uav/blob/main/graphical_abstract.png)

## What does this repository contain?

The repository is organized into a three-stage pipeline:

- **src/extract/** — Data extraction: reads orthophotos and DEMs, calculates viewing/solar angles, merges pixel-level data
- **src/filter/** — Spectral filtering: applies vegetation indices (OSAVI, Excess Green) and splits data by polygon
- **src/modelling/** — RPV model fitting: fits Rahman-Pinty-Verstraete models per plot and week
- **src/stats/** — Statistical analysis: ANOVA, logistic regression, and effect size calculations
- **src/core/** — Shared utilities: configuration, logging, preprocessing, and file search

## How to use this method to unlock multi-angular reflectance data?

### Installing required software

**Python:** Python 3.10 or higher is required. We recommend using a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate  # Windows
```

**Agisoft Metashape:** Please download [Agisoft Metashape Professional](https://www.agisoft.com/downloads/installer/) and purchase a license to allow full functionality.

**Python dependencies:** Install all required libraries:

```bash
pip install -r requirements.txt
pip install -e .
```

### Sample Coordinates

For each ground sampling that was performed in the field, you will need an associated coordinate and provide a csv file containing these coordinates.

NO HEADER
"id_1","lon_1","lat_1"
"id_2","lon_2","lat_2"
"id_i","lon_i","lat_i"

### Metashape

1. Align image dataset
2. Build dense cloud
3. Build DEM
4. Build orthomosaic
5. Export DEM (make sure the DEM matches the extent of the Orthomosaic)
6. Export orthomosaic (make sure the Orthomosaic matches the extent of the DEM)
7. Export orthophotos
8. Export camera positions as omega_phi_kappa.txt

### Combine required files into a single directory

Please save the following files in a single directory:

- DEM (as .tiff)
- Orthophotos (as directory)
- Camera positions (as .txt)
- Sample coordinates (as. csv)
- Original images (as directory)

### Python

1. Please download the complete [repository](https://github.com/ReneHeim/proj_on_uav) and keep the directory structure as it is
2. Copy and edit the configuration file:
   ```bash
   cp src/config_file_example.yml my_config.yml
   ```
3. Change the paths, settings, and output in `my_config.yml` according to your specific setup
4. Run the three pipeline stages:
   ```bash
   # Step 1: Extract per-pixel reflectance and geometry
   python -m src.pipeline_extract_data --config my_config.yml

   # Step 2: Apply spectral filters and split by polygon
   python -m src.pipeline_filtering --config my_config.yml

   # Step 3: Fit RPV models
   python -m src.pipeline_modelling --config my_config.yml --band band1
   ```
5. Alternatively, use the Makefile shortcuts:
   ```bash
   make extract
   make filter
   make rpv
   ```

### Contact

If you have any questions how to use the code, please commit an issue for others to benefit from it. If this is not an option for you, please contact either Nathan Okole (okole@ifz-goettingen.de) or Rene Heim (rheim@uni-bonn.de)


```{tableofcontents}
```
