# Script Registry

This registry documents the reorganized script namespace. Status values:

- `production`: primary command path for current workflows.
- `analysis`: scientific analysis used for figures, tables, or reports.
- `diagnostic`: QA, debugging, product comparison, or calibration checks.
- `archived`: historical one-off or superseded script retained for provenance.

| Status | Script Area | Purpose | Expected Outputs |
|---|---|---|---|
| production | `preprocessing/rededgep/` | RedEdge-P calibration, alignment, stack writing, and week orchestration | ONCERCO processing folders plus `outputs/preprocessing/rededgep/` reports/manifests |
| production | root wrappers | Backward-compatible entrypoints for active RedEdge-P commands | Delegates to `preprocessing/rededgep/` |
| diagnostic | `diagnostics/alignment/` | CPU/GPU SIFT alignment QA and benchmarks | `outputs/quality/`, `outputs/cpu_sift_runs/`, `outputs/logs/` |
| diagnostic | `diagnostics/metashape_compatibility/` | Metashape/custom radiometry and visual comparisons | `outputs/reports/metashape_custom_matchtest/`, `outputs/figures/metashape_custom_visual_diagnostics/` |
| analysis | `analysis/early_warning/` | Early-warning disease prediction experiments | `outputs/early_warning_severity_2024/`, cross-year outputs |
| analysis | `analysis/severity/` | Severity prediction, residual pipeline, cultivar checks | `outputs/cross_year_generalization_2024_to_2025/`, `outputs/multiangular_distribution_feature_family/` |
| analysis | `analysis/canopy_structure/` | LAI/LIA/canopy gap reflectance analyses | `outputs/backup_metadata/`, `outputs/result_02_canopy_gap_vza/` |
| analysis | `analysis/sun_geometry/` | VZA/RAA/phase angle and weather analyses | `outputs/result_01_raa_sun_geometry/`, `outputs/result_03_vza_curve_shape_metrics/` |
| analysis | `analysis/manuscript_tables/` | Paper-ready tables and summaries | `outputs/manuscript_tables/`, manuscript reports |
| analysis | `analysis/eda/` | Exploratory metadata/reflection links | `outputs/backup_metadata/eda*` |
| production | `metadata/` | Backup metadata extraction | `outputs/backup_metadata/` |
| utility | `plotting/` | Plot-only scripts for selected model summaries | analysis-family figure folders |
| archived | `archive/legacy_week1_gpu/` | Historical week1 GPU pipeline and inspection scripts | existing logs and external week1 products |
| archived | `archive/one_off_repairs/` | One-time band swap/repair scripts retained for audit | existing repair manifests/logs |

When adding a new script, place it in a purpose folder and update this file with its status,
purpose, and output root.
