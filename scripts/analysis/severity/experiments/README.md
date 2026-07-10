# Severity Experiment Taxonomy

The maintained cross-year, compact-distribution, and frozen residual workflows
remain directly under `scripts/analysis/severity/`. Methodological experiments
are grouped here by the question they test:

- `curve_shape/`: whether angular reflectance-curve shape adds information.
- `geometry/`: VZA, RAA, geometry fusion, and healthy-reference hypotheses.
- `sampling/`: whether spatial subplots change the unit of analysis.
- `comparisons/`: feature-selection, cultivar, and transfer sensitivity checks.

Legacy module paths remain relative symlinks so earlier reports and imports stay
reproducible. New reusable logic must be extracted to `src/research/severity/`.
