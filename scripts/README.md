# Scripts Layout

The `scripts/` directory is organized by purpose. Root-level Python files are compatibility
wrappers only; new implementation code should live in a subfolder.

## Production Entry Points

Use these stable commands:

```bash
python -m scripts.run_2025_rededgep_preprocess --week week2
python -m scripts.micasense_rededgep_preprocess --help
```

The actual implementations live in:

```text
scripts/preprocessing/rededgep/
```

## Folder Taxonomy

| Folder | Purpose |
|---|---|
| `preprocessing/rededgep/` | MicaSense RedEdge-P calibration, alignment, stack writing, and week orchestration |
| `preprocessing/orthorectification/` | Per-capture orthorectification and ODM-style product preparation |
| `preprocessing/odm/` | ODM installation/run helpers |
| `analysis/early_warning/` | Early-warning disease prediction experiments |
| `analysis/severity/` | Severity prediction, residual models, cultivar checks, and bottleneck studies |
| `analysis/canopy_structure/` | LAI/LIA/canopy openness and VZA reflectance studies |
| `analysis/sun_geometry/` | VZA, RAA, phase-angle, and weather/angular diagnostics |
| `analysis/manuscript_tables/` | Paper-ready tables and report summaries |
| `analysis/eda/` | Exploratory links between backup metadata and reflectance |
| `diagnostics/metashape_compatibility/` | Metashape/custom product comparisons and radiometry tests |
| `diagnostics/alignment/` | SIFT/GPU/CPU alignment diagnostics and QA |
| `metadata/` | Metadata extraction from backup spreadsheets |
| `plotting/` | Plot-only scripts reused by reports |
| `archive/` | Historical one-off scripts kept for provenance |

## Rules For New Scripts

- Prefer `python -m scripts.<subfolder>.<module>` execution.
- Write logs to `outputs/logs/` or the analysis-specific `logs/` folder.
- Write a markdown summary with exact output paths.
- Keep useful debug/profiling logs; remove dead code and stale commented experiments.
- Do not introduce new hardcoded `/mnt/data` paths when a CLI argument can be used.
- RedEdge-P stack order must be `Blue, Green, Red, Red edge, NIR`.
- RedEdge-P stack scale is `reflectance = uint16 / 32767`.
- Full per-capture panel detection is slow. The generic RedEdge-P runner defaults to
  `--panel-strategy none`; use `--panel-strategy full` only when diagnostic panel filtering
  is required.
