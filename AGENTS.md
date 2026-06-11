# AGENTS.md

## Project Objective

This repository works with the **ONCERCO multiangular drone imagery dataset**.

The data consist of multiangular drone images that have been orthorectified and are stored outside the repository in the Heim mounted folder:

```bash
/run/media/davidem/Heim/
```

Do **not** add raw images or large derived image products to Git.

## Data Context

The project includes multiple drone camera flights acquired over different weeks in the field.

Each field contains different polygons. Each polygon is an **AOI**.

Each AOI/polygon has associated field-trial metadata:

* polygon boundary
* field
* week or flight date
* cultivar
* treatment status
* disease observation, when available

Each polygon contains a specific cultivar and can be treated or untreated with an anti-pest or anti-disease agent.

## Repository Goal

The objective of this repository is to collect data from the drone images and preprocess them into a correct schema for reflectance analysis.

The repository should support:

* loading data from `/run/media/davidem/Heim/`
* linking images to fields, weeks, polygons, cultivars, treatments, bands, and viewing angles
* extracting reflectance values from AOIs
* filtering and aggregating per-pixel data
* preparing clean tables for reflectance analysis
* comparing nadir-only data with multiangular data

## Research Goal

The scope of the paper is to test whether using different viewing angles improves disease prediction before the disease becomes too severe or visually obvious.

The paper should evaluate the value of **multiangular data versus nadir-only data** for disease prediction or prevention.

The analysis should use different spectral bands from the MicaSense sensor.

Agents should check the notebooks in `docs/` for additional research context.

## Scientific Motivation

Sugar beet reflectance is strongly affected by leaf angle and canopy structure.

Because of this, multiangular imagery has a large effect on the spectral signature of the plants.

The project currently uses the RPV model to model angular reflectance, but the RPV results show poor values.

The RPV model does not reliably model the reflectance of the plants in this dataset.

The likely reason is that sugar beet leaf angle strongly affects the reflectance profile. This makes the angular signal complex and difficult to model with a simple RPV approach.

## Interpretation of RPV Results

Do not assume that RPV failure means the multiangular data are useless.

Instead, treat the poor RPV fit as evidence that sugar beet canopy reflectance is complex and strongly angle-dependent.

The key idea is:

```text
Multiangular reflectance should not be discarded as noise.
It may contain useful information for disease detection.
```

## Paper Framing

The paper should try to provide enough evidence for publication showing that multiangular drone data has value for disease prediction.

The central argument is:

```text
Standard nadir or orthomosaic workflows may discard useful off-nadir information.
This repository investigates whether that multiangular information improves sugar beet disease prediction compared with nadir-only data.
```

## Main Research Question

```text
Can multiangular MicaSense drone imagery improve early sugar beet disease prediction compared with nadir-only imagery?
```


## Agent Instructions

### Mandatory rules for all processing/analysis scripts

**Rule 1 — Function-level profiling and log files**

Every data processing or analysis script MUST:

1. Time every major phase (data loading, preprocessing, feature building, model fitting, aggregation) using `time.time()` or `time.perf_counter()`.
2. Print per-phase timings to stdout AND write them to a structured log file in `outputs/logs/{script_name}_{timestamp}.log`.
3. Use the pattern:
   ```python
   t0 = time.time()
   # ... work ...
   logging.info(f"[PHASE] name: {time.time() - t0:.1f}s")
   ```
4. For I/O-heavy operations (parquet loading), profile individual file read times and report summary (min/median/max/mean).
5. For ML operations, profile fit vs predict separately.

Bottlenecks must be identifiable from the log alone without re-running the script.

**Rule 2 — Markdown results summary for paper reporting**

Every analysis script MUST produce a markdown file in `outputs/reports/{script_name}_summary.md` containing:

1. A **markdown table** with the key results (numbers, metrics, p-values) — ready to copy into the paper.
2. A **1-2 sentence interpretation** of what the results mean for the research question.
3. The **exact file paths** of all outputs produced.
4. A **reproducibility section** listing the config, random seed, cross-validation splits, and any parameters used.

Example format:
```markdown
## Results: Feature Set Comparison

| Feature Set | Mean AUROC | Std AUROC | Type |
|------------|-----------|-----------|------|
| M3 (VZA)   | 0.717     | 0.072     | Multiangular |
| M1 (Nadir) | 0.581     | 0.034     | Nadir-only |

**Interpretation**: Multiangular features (M3) improved AUROC by +0.136 over nadir-only (M1),
suggesting that off-nadir viewing angles carry disease-relevant reflectance information.

**Outputs**: `outputs/results/model_comparison_by_fold.csv`
**Config**: `configs/paths.yaml`, seed=42, StratifiedGroupKFold(n_splits=5)
```

---

### How to run the pipeline

Use `python -m src.pipeline_*` — **not** `python src/pipeline_*.py` (import paths resolve correctly with `-m`).

```bash
# Step 1: Extract per-pixel reflectance + angles + polygon filtering
python -m src.pipeline_extract_data --config my_config.yml
# Add --no-polygon to skip polygon filtering (data won't have plot_id)
# Add --alignment to co-register DEM/orthophoto if misaligned

# Step 2: Compute OSAVI/ExcessGreen, split into per-plot parquets
python -m src.pipeline_filtering --config my_config.yml

# Step 3: Fit RPV models + stats (ANOVA, logistic regression)
python -m src.pipeline_modelling --config my_config.yml --band band1
# Use --band 0 for all 5 bands
# Use --base-dir to point to the data root with per-plot parquets
```

Or via Makefile: `make extract`, `make filter`, `make rpv`

### Test commands

```bash
python -m pytest tests/ -v              # all tests
python -m pytest tests/ -q --tb=line    # fast, compact
python -m pytest tests/ --durations=20  # find slow tests
```

190+ tests, all should pass. No GUI windows pop up (conftest.py sets Agg backend).

### Key paths on the Heim drive

```
/run/media/davidem/Heim/
├── 2024/                    # 2024 season
│   ├── 20240603_week0/      # extraction: 1110 parquets ✓ full
│   ├── 20240611_week1/      # orthophotos are 3-band only (RGB) ✗
│   ├── 20240624_week3/      # extraction: 153 parquets ✓ full
│   ├── 20240715_week5/      # extraction: 283 parquets ✓ full
│   ├── 20240723_week6/      # no camera/DEM available ✗
│   └── 20240826_week8/      # extraction: 563 parquets ✓ full
├── 2025/                    # 2025 season
│   ├── week0/               # extraction: 286 parquets ✓ full
│   ├── week3/               # extraction: 274 parquets ✓ full
│   └── week5/               # extraction: 101 parquets ✓ full
├── 2024_oncerco_plot_polygons.gpkg   # 24 plots: columns [cult, ifz_id, trt, ino, geometry]
├── 2025_oncerco_plot_polygons.gpkg   # 24 plots: columns [cultivar, trt, geometry] — NO ifz_id!
├── RPV_Results/             # RPV outputs (V1 through V12)
└── stats/                   # Stats outputs (V1)
```

**Note**: 2025 polygon file lacks `ifz_id` column. The modelling pipeline now auto-generates sequential IDs. 2024 wk1 has only 3-band RGB — pipeline expects 5 bands.

### Configuration pattern

Config files use `{base_path}` substitution. Required structure:

```yaml
base_path: "/run/media/davidem/Heim/2024/20240624_week3/metashape/20241206_week3_products_uav_data"
inputs:
  date_time:
    start: "2024-06-24 12:00:00"
    time_zone: "Europe/Berlin"
  paths:
    cam_path: "{base_path}/20241206_week3_cameras.txt"
    dem_path: "{base_path}/20241206_week3_dem.tif"
    orthophoto_path: "{base_path}/20241212_week3_orthophotos/*.tif"
    ori: ["{base_path}/20241212_week3_orthophotos"]
    polygon_file_path: "/run/media/davidem/Heim/2024_oncerco_plot_polygons.gpkg"
  settings:
    bands: 5
    target_crs: "EPSG:32632"
outputs:
  paths:
    main_out: "{base_path}/output"
    plot_out: "{base_path}/output/plots"
```

### Common issues and fixes

1. **`python src/pipeline_*.py` fails** with `ModuleNotFoundError: No module named 'src'` → use `python -m src.pipeline_*` instead
2. **Missing `plot_id` in parquets** → extraction was run without polygon filtering. Re-run without `--no-polygon`.
3. **`ifz_id` column not found** → 2025 polygon file uses different schema. Fixed in pipeline (auto-generates IDs).
4. **Plots pop up during extraction** → fixed. All `plt.show()` calls in extraction path are now save-only.
5. **Modelling crashes on stats phase** → the logistic regression may fail on edge data. RPV fitting still works.
6. **Filter pipeline finds "1 files lack plot_id"** → that file was extracted without polygons. It's skipped. Re-extract if needed.
7. **Some weeks show partial extraction** → images outside polygon boundaries are correctly skipped (no plot overlap).

### Validation

Post-extraction validation runs automatically. Also available standalone:

```bash
python -m src.core.validate --dir /path/to/extract/output
```

Checks: schema consistency, required columns, VZA range (0-90°), band value ranges.

### Bands and sensor (MicaSense Altum)

| Band | Wavelength | Name |
|------|-----------|------|
| band1 | 475 nm | Blue |
| band2 | 560 nm | Green |
| band3 | 668 nm | Red |
| band4 | 717 nm | Red Edge |
| band5 | 842 nm | NIR |

OSAVI uses band3 (red) and band5 (NIR): `(1+0.16)*(NIR-Red)/(NIR+Red+0.16)`

### Viewing angles available

| Parameter | Range | Notes |
|-----------|-------|-------|
| VZA (view zenith) | 0°–73° across images | Per-image spread: ~20° median |
| VAA (view azimuth) | 0°–360° full circle | Full azimuthal coverage from flight grid |
| SZA (sun elevation) | 25°–87° | Varies by time of day |
| RPV bins | [0, 15, 25, 35, 45, 60] deg | All bins populated |

### Architectural notes

- **Polars** for all dataframes (not pandas). Use `pl.read_parquet`, `pl.DataFrame`, lazy `pl.scan_parquet`
- **Import path**: all src imports use `from src.X.Y import Z` (absolute, src-prefixed)
- **Entry points**: `main_extract.py`, `filtering.py`, `rpv_modelling.py` at project root are thin wrappers using `importlib` for CLI
- **Resume support**: extraction pipeline skips already-processed images by checking parquet filenames in output dir
- **Plotting**: matplotlib Agg backend in tests. Source functions save to file when `output_path` is given, never `plt.show()` unconditionally
- **Polygon filtering**: parallelized via `process_chunks_parallel` with ThreadPoolExecutor (configurable via `number_of_processor`)

## Expected Analysis Outputs

### Extraction (`pipeline_extract_data.py`)
- One `.parquet` per orthophoto in `output/`
- Columns: Xw, Yw, band1-5, elev, vza, vaa, sunelev, saa, delta_z, delta_x, delta_y, distance_xy, path, [plot_id]
- Validation report at end of run (schema + range checks)

### Filtering (`pipeline_filtering.py`)
- One `.parquet` per polygon in `output/plots/plot_{id}.parquet`
- Adds OSAVI and ExcessGreen columns
- Files missing `plot_id` are skipped (logged as "X files lack plot_id")

### Modelling (`pipeline_modelling.py`)
- `RPV_Results/V12/{week}/rpv_{week}_{band}_results.csv` per week per band
- Aggregate `rpv_results.csv` with all weeks/bands combined
- RPV parameters: rho0, k, theta, rc, rmse, nrmse
- Status column: "success" or "error: ..."
- Stats: `stats/V1/{week}/` with ANOVA results, KDE plots, logistic regression outputs

### Data quality expectations

- **NRMSE**: 0.44–0.69 for RPV fits (high, reflects model limitations with sugar beet)
- **rho0**: ~0.03 (blue) to ~0.27 (NIR) for healthy vegetation
- **theta**: near 0 ± 0.2 (low BRDF asymmetry captured)
- ~25% of orthophotos overlap field plots (remaining are edge/routing images)


