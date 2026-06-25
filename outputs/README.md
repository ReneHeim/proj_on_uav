# Outputs Layout

Historical outputs are intentionally left in place. New outputs should follow a predictable
layout so each result can be traced back to the script that created it.

## Preferred New Layout

```text
outputs/
├── preprocessing/
│   └── rededgep/{week}/
│       ├── reports/
│       ├── results/
│       ├── figures/
│       ├── logs/
│       └── manifests/
├── analysis/{analysis_family}/
│   ├── reports/
│   ├── results/
│   ├── figures/
│   ├── logs/
│   └── manifests/
├── manuscript/
│   ├── tables/
│   ├── figures/
│   └── reports/
├── diagnostics/{diagnostic_family}/
└── archive/
```

## Provenance Rules

- Every script should write a markdown summary with exact output paths.
- Every processing/analysis run should write a structured log.
- Long-running preprocessing should write a manifest containing inputs, settings, output root,
  selected captures, failures, and band-order metadata.
- Existing historical folders are indexed rather than moved during the first refactor.
