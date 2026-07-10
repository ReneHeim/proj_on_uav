# Outputs Layout

`outputs/` stores ignored local research artifacts. The canonical hierarchy is:

```text
outputs/
├── shared/                    # reusable labels and derived feature tables
├── runs/
│   ├── preprocessing/
│   ├── analysis/
│   │   ├── severity/{current,future,cross_year,experiments}/<run>/
│   │   ├── early_warning/
│   │   ├── reflectance/
│   │   ├── canopy_structure/
│   │   └── sun_geometry/
│   ├── diagnostics/
│   └── metadata/
├── deliverables/
│   ├── manuscript/
│   └── presentation/
├── archive/
│   ├── historical_runs/
│   └── legacy_unscoped/
└── provenance/
```

Every new run must contain `results/`, `figures/`, `reports/`, `logs/`, and
`manifests/`. Use `src.research.common.RunPaths` to create this structure.

The old-to-new mapping is recorded in `outputs/provenance/output_layout_migration_20260710.csv`.
There are no compatibility symlinks: maintained code must use canonical paths.

## Provenance Rules

- Every script writes a markdown summary with exact output paths and its input artifact IDs.
- Every processing or analysis run writes a structured log within its own `logs/` directory.
- Long-running preprocessing writes a manifest with inputs, settings, output root, selected captures,
  failures, and band-order metadata.
- New maintained workflows resolve prerequisite artifacts from `configs/outputs.yaml`, not from hardcoded
  output paths.
