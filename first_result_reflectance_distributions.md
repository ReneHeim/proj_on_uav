# Result 1 Plan: Reflectance Distributions Across Viewing Angles

## Research Question

How does sugar beet canopy reflectance change across viewing angles for each MicaSense spectral band?

This first result establishes whether the dataset contains a systematic angular reflectance signal before testing its relationship with disease. It should show whether nadir and off-nadir observations provide measurably different descriptions of the same canopies.

## Hypothesis

Reflectance distributions will differ across view-zenith-angle (VZA) classes, with the largest angular responses expected in the red edge and near-infrared bands because these bands are strongly influenced by canopy structure, leaf orientation, and shadowing.

## Selected Season

Use **2024** as the primary year for this result. It has the longest temporal sequence, established plot metadata, field measurements, and existing processed multiangular parquets for weeks 0, 3, 5, and 8. The analysis must show both the angular reflectance distribution within each week and how that distribution changes over the season.

The primary temporal questions are:

- Does the reflectance distribution within each VZA class change from week to week?
- Does the difference between nadir and off-nadir reflectance become larger or smaller as the canopy develops?
- Are temporal changes consistent across the five spectral bands?
- Do cultivars differ in their angular reflectance distributions or in how those distributions change over time?
- Are changes in the angular distribution associated with canopy development or later disease progression?

## Self-Contained Result Workspace

Keep all code and manageable outputs for this analysis in one self-contained folder inside the analysis repository:

```text
/home/davidem/PycharmProjects/proj_on_uav/outputs/result_01_reflectance_distributions/
├── README.md
├── configs/
├── manifests/
├── 2024/
│   ├── unfiltered/
│   │   ├── results/
│   │   ├── figures/
│   │   │   ├── main/
│   │   │   └── supplementary/
│   │   ├── reports/
│   │   └── logs/
│   └── ground_filtered/
│       ├── results/
│       ├── figures/
│       ├── reports/
│       └── logs/
└── 2025/
    ├── unfiltered/
    └── ground_filtered/
```

This folder should contain everything needed to understand and reproduce Result 1 without searching across unrelated output directories:

- the exact analysis configuration;
- input and recovery manifests;
- compact aggregated CSV/parquet tables;
- model-result tables;
- final and supplementary figures;
- profiling logs;
- the Markdown results summary;
- a `README.md` describing how the result was generated and where large external data are stored.

Do **not** copy raw images, orthophotos, DEMs, Metashape projects, per-pixel extraction parquets, or other large products into the Git repository. These must remain on writable `/run/media/davidem/Heim/`. The self-contained result folder should reference them through documented absolute paths and a machine-readable manifest.

To prevent repository bloat:

- keep only aggregated plot-week-angle tables in the result folder;
- store large newly extracted per-pixel and per-plot parquets under `/run/media/davidem/Heim/2024/recovered_weeks/`;
- do not duplicate the same output in both the shared `outputs/` hierarchy and the Result 1 folder;
- use compressed parquet for analytical tables when CSV would be unnecessarily large;
- keep only final figures and essential diagnostic figures;
- exclude temporary caches, intermediate chunks, raster products, and debug exports through `.gitignore`;
- document every external file in `manifests/external_data_manifest.csv`, including source, size, checksum when practical, role, and read/write status.

The result folder should be relocatable as a reporting package, except for the explicitly documented external Heim data dependencies.

Current generated analysis outputs are organized by season and filter state:

```text
outputs/result_01_reflectance_distributions/<year>/<filter_state>/
```

where `<year>` is `2024` or `2025`, and `<filter_state>` is:

- `unfiltered`: valid positive-reflectance observations after VZA and band-quality filters;
- `ground_filtered`: the same workflow after removing likely ground/background pixels using `OSAVI > 0.2` before plot-level aggregation.

## Data Required

Use the processed per-pixel parquet files in the active workspace at `/run/media/davidem/Heim/`. For weeks without parquet extraction, use the existing Metashape products in the active workspace or read them from `/run/media/davidem/heim_data/Backup/proj_on_cerco/`. The backup is strictly read-only. All new parquets, intermediate products, logs, configurations, validation reports, and figures must be written under `/run/media/davidem/Heim/` or the paper/repository output folders.

Required columns:

- `plot_id`
- acquisition week and year
- cultivar
- `band1` to `band5`
- `vza`
- `vaa`
- reflectance quality or vegetation-mask columns, when available

Band interpretation:

| Column | Wavelength | Band |
|---|---:|---|
| `band1` | 475 nm | Blue |
| `band2` | 560 nm | Green |
| `band3` | 668 nm | Red |
| `band4` | 717 nm | Red edge |
| `band5` | 842 nm | NIR |

## Viewing-Angle Classes

Use the existing VZA boundaries to keep this result consistent with the other analyses:

| Class | VZA range |
|---|---|
| Nadir | 0--15 degrees |
| Low off-nadir | 15--25 degrees |
| Moderate off-nadir | 25--35 degrees |
| High off-nadir | 35--45 degrees |
| Very high off-nadir | 45--60 degrees |
| Extreme off-nadir | 60--90 degrees |

Report the number of plots, images, and pixels in every class. Sparse classes must be identified and should not be interpreted as strongly as well-supported classes.

## Analysis Steps

### 1. Extract Missing 2024 Weeks from Existing Metashape Products

The currently extracted parquet series contains weeks 0, 3, 5, and 8. Inspection confirmed that the other weeks are not generally missing Metashape processing; most already have Metashape projects and exported products. What is missing is the current repository's per-pixel and per-plot parquet extraction.

Verified product inventory:

| Week | Metashape project | Camera export | DEM | Per-image orthophotos | Sample band structure | Current parquet extraction |
|---:|---|---|---|---:|---|---|
| 1 | Available | Available | Available | 580 | 3-band Byte/RGB | Missing; exclude from five-band analysis |
| 2 | Available | Available | Available | 312 | 5-band Float32 | Missing; ready for extraction |
| 4 | Available | Available | Available | 322 | 5-band Float32 | Missing; ready for extraction |
| 6 | Available | Available | Available | 325 | 5-band Float32 | Missing; ready for extraction from active Heim |
| 7 | Available | Available | Available | 322 | 5-band Float32 | Missing; ready for extraction |

Weeks 2, 4, and 7 have their Metashape products in the read-only backup. Week 6 already has its Metashape project, camera export, DEM, orthomosaic, and per-image five-band orthophotos under active `/run/media/davidem/Heim/2024/20240723_week6/`. Therefore, Metashape reprocessing should not be performed unless validation discovers that an exported product is unusable.

For every missing parquet week, verify the inputs required by the current extraction pipeline:

- complete five-band RedEdge-P imagery;
- camera positions and orientations, or a recoverable Metashape camera export;
- a DEM or sufficient source products to create one;
- orthophotos or the inputs required to generate them;
- acquisition time and coordinate reference information;
- the 2024 plot polygon layer.

Process weeks 2, 4, 6, and 7 because all four have five-band per-image orthophotos, camera exports, and DEMs. Week 1 must not be combined with the five-band result because its available per-image orthophotos are three-band Byte/RGB products.

Read source material directly from the backup when technically safe, or copy only the required metadata/configuration into the active workspace. Never create temporary files, Metashape products, corrected images, logs, or exports inside the backup.

Write recovered data to a dedicated active-workspace structure such as:

```text
/run/media/davidem/Heim/2024/recovered_weeks/week2/output/
/run/media/davidem/Heim/2024/recovered_weeks/week2/output/plots/
/run/media/davidem/Heim/2024/recovered_weeks/week4/output/
/run/media/davidem/Heim/2024/recovered_weeks/week4/output/plots/
/run/media/davidem/Heim/2024/recovered_weeks/week6/output/
/run/media/davidem/Heim/2024/recovered_weeks/week6/output/plots/
/run/media/davidem/Heim/2024/recovered_weeks/week7/output/
/run/media/davidem/Heim/2024/recovered_weeks/week7/output/plots/
```

For weeks 2, 4, and 7, configure the pipeline to read the existing Metashape products directly from the read-only backup while writing every output to active Heim. For week 6, read the products from active Heim and write the extraction to the dedicated recovery output directory. Run extraction, filtering, and validation through the repository entry points using `python -m src.pipeline_*`. Record source paths, destination paths, processing configuration, band count, image count, output parquet count, plots covered, and validation results. A newly extracted week may enter the analysis only when its schema and reflectance/angle ranges match the existing 2024 data.

### 2. Prepare and Validate the Combined 2024 Data

Combine the validated existing and newly extracted 2024 per-plot parquets. Retain valid vegetation reflectance observations and remove invalid or physically implausible values according to the project validation rules. Record how many rows are removed and why.

Produce a week-availability table before analysis:

| Week | Source | Processing status | Bands | Plots | Pixels | Included in analysis | Exclusion reason |
|---:|---|---|---:|---:|---:|---|---|

Only comparable five-band weeks should enter the main result. Incomplete weeks must remain visible in this table, with a precise reason for exclusion.

The analysis script must profile individual parquet reads and report the minimum, median, mean, and maximum read times in its log.

### 3. Create Balanced Plot-Level Summaries

For every 2024 week, plot, VZA class, and spectral band, calculate:

- number of pixels;
- mean reflectance;
- median reflectance;
- standard deviation;
- interquartile range;
- 5th and 95th percentiles.

The plot-week-angle summary should be the main analytical unit. Raw pixels are not independent replicates, so statistical interpretation must not treat billions of pixels as billions of independent samples.

### 4. Produce the Main Distribution Figures

Create one multi-panel figure with one panel per spectral band. Within each panel, show the distribution of plot-level median reflectance for every VZA class using violin or box plots with the individual plot values visible.

Use the same angle colors and ordering in every panel. Display the number of plot-week observations beneath or above each angle class.

Create a second main figure showing the temporal development of the distributions. Use one row per spectral band and one column per included week, or one panel per band with week on the horizontal axis and separate distributions or lines for each VZA class. Keep the reflectance scale comparable across weeks within each band.

This temporal figure must make it possible to see whether the full distribution shifts, widens, narrows, or changes shape during the season, rather than showing only a change in the mean.

### 5. Produce Angular Reflectance Curves

For each band, plot mean or median reflectance against the center of each VZA class. Show the overall curve with a 95% confidence interval and faint lines for individual plots or plot-week observations.

Create week-specific curves as a central result, followed by supplementary versions separated by:

- cultivar, as a diagnostic;
- treatment, as a diagnostic;
- individual plot, for selected representative examples.

The week-specific curves will reveal how the angular response develops over time. The diagnostic figures will show whether the temporal pattern is stable or driven by one cultivar, treatment, or small group of plots.

### 6. Quantify Angular and Temporal Change

For every band and off-nadir class, calculate both the absolute and relative contrast against the matched nadir observation from the same plot and week:

```text
absolute contrast = off-nadir reflectance - nadir reflectance
relative contrast = (off-nadir reflectance - nadir reflectance) / nadir reflectance
```

Report the median contrast and a 95% confidence interval across matched plot-week observations. This directly quantifies how much information changes when moving away from nadir.

Also calculate matched temporal changes for plots present in consecutive included weeks:

```text
temporal change = reflectance at later week - reflectance at earlier week
change in angular contrast = angular contrast at later week - angular contrast at earlier week
```

These quantities distinguish a general seasonal reflectance change from a change specifically affecting the multiangular signal.

### 7. Test Cultivar Differences

Compare the angular reflectance response between the sugar beet cultivars represented in the 2024 trial. This is important because cultivar-specific leaf angle, canopy structure, pigmentation, and growth development may alter the measured reflectance independently of disease.

For every band, week, and VZA class, summarize the plot-level reflectance distribution separately by cultivar. Compare both the absolute reflectance and the matched off-nadir minus nadir contrast. Use plots as the biological replicates and report the number of plots available for each cultivar-week combination.

Test the following effects with a repeated-measures or mixed-effects model:

```text
reflectance ~ cultivar * VZA_class * week + treatment + (1 | plot_id)
```

If the sample size does not support the full three-way interaction, use a reduced model and test cultivar-by-angle and cultivar-by-week interactions separately. Report effect sizes and confidence intervals, not only p-values.

The key cultivar questions are:

- Do the cultivars have different overall reflectance levels?
- Does the angular reflectance curve differ between cultivars?
- Does one cultivar show a stronger off-nadir response?
- Do cultivar differences change as the canopy develops across weeks?
- Could cultivar-specific canopy architecture explain part of the later disease-related angular signal?

Cultivar must not be interpreted as a disease effect. Later disease analyses should either control for cultivar, stratify by cultivar, or demonstrate that the multiangular disease result is present within both cultivars.

### 8. Check Directional Effects

VZA alone does not describe the full viewing geometry. As a supporting analysis, inspect whether reflectance at a similar VZA also changes with view azimuth or relative azimuth to the sun. This check will determine whether a VZA-only summary hides important directional effects.

### 9. Planned Extensions and Sensitivity Tests

After the primary 2024 Result 1 analysis is stable, run the same descriptive statistics and figure-generation workflow for the validated 2025 weeks. The 2025 output should mirror the 2024 result structure as closely as possible so the two seasons can be compared directly. Produce the same reflectance-by-VZA summaries, matched angular contrasts, angular reflectance curves, cultivar or treatment diagnostics when metadata allow, and paper-ready figures. Keep the 2025 analysis clearly labeled as a cross-season validation or extension rather than mixing it into the primary 2024 result without a separate justification.

Test whether adding a ground or bare-soil filter changes the angular reflectance signal. The filtered analysis should use the same VZA bins, plot-level aggregation, statistics, and figures as the main result, but should exclude likely ground/background pixels before aggregation. Record the exact filtering rule, thresholds, affected row counts, affected plot-week-angle counts, and whether the direction or magnitude of the off-nadir contrasts changes after filtering.

Analyze directional illumination effects using view azimuth angle and relative azimuth angle to the sun. Where possible, derive relative azimuth from `vaa` and solar azimuth (`saa`) and compare reflectance within matched or narrow VZA bins. This should test whether the angular signal differs between sun-facing, cross-sun, and backscatter-like viewing directions. Report the available support per RAA class because sparse azimuth coverage should limit interpretation.

## Figures and Tables

## Figure Design System

The figures should look like a coherent scientific series rather than unrelated default plots. Use the same visual language throughout Result 1 so readers can compare bands, weeks, angles, and cultivars without relearning the design.

### General Style

- Use a clean white background with very light horizontal reference lines only. Remove top and right axes and avoid full rectangular plot borders.
- Use a publication font consistently, preferably Arial, Helvetica, or the journal's final sans-serif font. Use approximately 8--9 pt text at final printed size, 9--10 pt axis labels, and 10--11 pt panel titles.
- Export vector versions as PDF or SVG and a 600 dpi TIFF version for submission. Figures must remain readable at the expected one- or two-column journal width.
- Use direct labels where space permits. Avoid repeatedly forcing the reader to move between the plot and a distant legend.
- Label panels with bold lowercase letters, such as `(a)` to `(e)`, in the upper-left corner at exactly the same position.
- Keep band names, units, VZA labels, cultivar names, and week notation identical across all figures.
- Use sentence case in titles and labels. Do not place a large title inside each figure; the caption should carry the full explanation.
- Do not use 3D charts, rainbow palettes, heavy grid lines, gradients, decorative shadows, or default software colors.
- Do not show p-value stars as the only statistical information. Prefer an effect-size estimate with a 95% confidence interval and add an exact p-value in a compact annotation or the caption.

### Color System

Separate the visual roles of spectral bands, viewing angles, and cultivars rather than assigning arbitrary colors in each figure.

**Spectral bands:** use physically meaningful, color-blind-conscious colors when each band is represented by a line:

| Band | Suggested color | Hex |
|---|---|---|
| Blue | clear blue | `#3B73B9` |
| Green | bluish green | `#2A9D6F` |
| Red | vermilion | `#D1495B` |
| Red edge | magenta | `#A64D79` |
| NIR | near-black violet | `#4B3F72` |

**Viewing angles:** when several VZA classes appear in the same band panel, use a sequential light-to-dark teal scale. Nadir should always be the darkest or visually strongest line, and progressively off-nadir classes should become lighter. This communicates angular order without using unrelated categorical colors.

**Cultivars:** use two clearly separated colors that are not already carrying the angle meaning:

- Cultivar 1: `#0072B2` (blue)
- Cultivar 2: `#D55E00` (orange/vermilion)

Use solid and dashed line styles in addition to color so cultivar comparisons remain readable in grayscale. Replace `Cultivar 1` and `Cultivar 2` with the verified cultivar names in the final plots.

### Figure 1: Seasonal Angular Distribution Atlas

This should be the main descriptive figure and the visual signature of Result 1.

- Layout: five rows for the spectral bands and one column for each included 2024 week.
- Each cell contains a compact half-violin plus box-and-point distribution of plot-level median reflectance across VZA classes.
- Use half-violins only when there are enough plot observations to estimate a distribution. Otherwise use a box plot with jittered plot points.
- Plot raw plot-level values as small semi-transparent points, not raw pixels. Connect the same plot across VZA classes with very thin, low-opacity lines in a supplementary version.
- Keep the same y-axis range across weeks within each spectral-band row. Different bands may use different ranges, but their scales must be printed clearly.
- Place week/date labels across the top and band names along the left edge. Avoid repeating axis labels inside every panel.
- Add a small `n =` annotation for the number of plots represented in each cell or VZA class.
- Mark unavailable or excluded weeks as deliberately empty cells with a short reason such as `3-band only`; do not silently remove them from the temporal sequence.

This atlas is preferable to a generic collection of violin plots because it exposes both the angular distribution and its seasonal evolution in one structured view.

### Figure 2: Angular Reflectance Fingerprints

Use line-and-ribbon small multiples to show the shape of the angular response.

- Layout: one panel per band, arranged as a 2-by-3 grid with the final cell used for a shared legend or concise interpretation key.
- Horizontal axis: VZA-bin midpoint in degrees, displayed on a true numeric scale rather than equally spaced category labels.
- Vertical axis: plot-level reflectance.
- Draw one line per week, ordered from a light early-season tone to a dark late-season tone. Add points at observed angle-bin centers.
- Add a narrow 95% bootstrap confidence ribbon around each weekly curve.
- Use direct labels at the final supported angle where curves are sufficiently separated. Otherwise use one compact legend ordered chronologically.
- Add faint individual plot curves only in a supplementary figure; the main figure should emphasize the population-level shape.
- Do not interpolate across unsupported angle bins. Break the line when a week lacks adequate plot support.

### Figure 3: Matched Off-Nadir Effect Plot

Show the estimated change relative to nadir instead of another raw distribution plot.

- Use a forest-plot layout with one row per band and off-nadir class.
- Horizontal axis: matched off-nadir minus nadir reflectance, preferably also presented as percentage change in a companion panel or table.
- Plot the median or model-estimated contrast as a point with a 95% confidence interval.
- Include a strong but thin vertical zero-reference line. Effects to the left and right should be immediately interpretable.
- Facet by week or use vertically aligned week columns. Keep the contrast scale fixed wherever scientifically reasonable.
- Print the number of matched plots beside each estimate in muted text.

This figure should carry the quantitative message more effectively than significance brackets over multiple violins.

### Figure 4: Cultivar Angular Response

Show whether cultivar changes the angular reflectance pattern without creating an unreadable interaction plot.

- Use rows for all five bands when space allows, because cultivar differences can also appear in blue and green through canopy cover, shadowing, and background exposure. If the all-band panel becomes too dense for the manuscript, keep all five bands as a supplementary figure and use red, red edge, and NIR in the main text.
- Use columns for weeks or disease-development stages.
- Draw one solid blue curve and one dashed orange curve for the two cultivars, with interquartile-range ribbons using low opacity.
- Add a narrow lower strip or adjacent forest plot showing the cultivar difference at each angle. This separates the observed curves from the inferential comparison.
- Use matched y-axis scales within each band across weeks.
- Annotate the cultivar-by-angle interaction estimate in the panel margin, not over the data.

### Figure 5: Seasonal Change in Angular Contrast

Show whether the multiangular signal itself changes during canopy development.

- Horizontal axis: week or acquisition date, with real temporal spacing if dates differ unevenly.
- Vertical axis: off-nadir minus nadir contrast.
- Use one panel per band and one line per off-nadir VZA class.
- Plot the matched plot-level observations faintly in the background and the median/model estimate prominently in front.
- Add confidence intervals and visually mark gaps caused by unavailable weeks.
- Consider a second aligned strip displaying DSDI severity or canopy-development measurements only as contextual information, without implying causation in Result 1.

### Figure Captions

Each caption should state:

1. the analytical unit, for example plot-week-angle median rather than pixels;
2. the angle-bin boundaries;
3. what the point, line, box, violin, or ribbon represents;
4. the number of plots and missing-data rule;
5. whether intervals are bootstrap confidence intervals or model-based intervals;
6. the exact comparison being made;
7. one restrained sentence describing the visible pattern, without claiming statistical significance unless it was tested.

### Accessibility and Integrity Checks

- Simulate deuteranopia and grayscale before final export.
- Confirm that line style, point shape, or direct labeling preserves cultivar and angle distinctions without color.
- Do not truncate axes in ways that exaggerate small reflectance differences. If a restricted axis is scientifically useful, state the range clearly and provide full-range supplementary plots.
- Keep plot ordering fixed: bands from blue to NIR, weeks chronologically, and angles from nadir to extreme off-nadir.
- Show uncertainty and sample size wherever a central estimate is shown.
- Do not smooth curves more strongly than the angular support permits.

### Main Figure

1. Five-panel violin or box plot of plot-level reflectance distributions across VZA classes, one panel for each spectral band.

2. Week-by-week distribution figure showing how each band and VZA class changes across the 2024 season.

3. Cultivar-specific angular reflectance curves showing the temporal response of each cultivar for all five bands.

### Main Table

| Week | Band | VZA class | Observations | Plots | Median reflectance | IQR | Change from nadir | Change from previous week |
|---:|---|---|---:|---:|---:|---:|---:|---:|

A second table should report cultivar differences:

| Week | Band | VZA class | Cultivar | Plots | Median reflectance | Angular contrast | Cultivar effect size | 95% CI |
|---:|---|---|---|---:|---:|---:|---:|---:|

### Supplementary Material

- Angular reflectance curves by week.
- Cultivar-specific distributions and angular reflectance curves.
- Cultivar-by-angle and cultivar-by-week model results.
- Matched temporal changes by band and angle.
- Plot-level matched off-nadir minus nadir contrasts.
- VZA coverage and sample-count table.
- View-azimuth diagnostic plots.
- Missing-week recovery and validation table.

## Interpretation Criteria

The result supports a meaningful angular signal if:

- reflectance changes consistently across VZA classes for matched plots;
- temporal changes are visible across weeks without being caused only by unequal plot coverage;
- the direction of angular change is reasonably stable, or its week dependence is clearly quantified;
- cultivar differences are quantified and are not incorrectly attributed to disease;
- confidence intervals for off-nadir contrasts indicate non-negligible differences from nadir;
- the pattern is not explained only by unequal plot coverage or one acquisition date.

A weak or inconsistent angular response is still informative. It may indicate strong dependence on illumination, canopy stage, azimuth, or disease state and would motivate the later interaction analyses rather than justify discarding multiangular data.

## Expected Paper Statement

Do not write the conclusion before the analysis is complete. The intended result statement should follow this structure:

> Reflectance varied systematically with viewing angle, with the magnitude and direction of the response differing among spectral bands. The matched plot-level analysis showed that off-nadir observations changed the measured canopy signal relative to nadir, providing the basis for testing whether these angular differences contain disease-relevant information.

## Required Outputs

The analysis should produce:

- `outputs/result_01_reflectance_distributions/<year>/<filter_state>/results/reflectance_by_vza_summary.csv`
- `outputs/result_01_reflectance_distributions/<year>/<filter_state>/results/matched_angular_contrasts.csv`
- `outputs/result_01_reflectance_distributions/<year>/<filter_state>/results/temporal_reflectance_changes_<year>.csv`
- `outputs/result_01_reflectance_distributions/<year>/<filter_state>/results/angular_contrast_changes_<year>.csv`
- `outputs/result_01_reflectance_distributions/<year>/<filter_state>/results/cultivar_angular_comparison_<year>.csv`
- `outputs/result_01_reflectance_distributions/<year>/<filter_state>/results/cultivar_angle_model_<year>.csv`
- `outputs/result_01_reflectance_distributions/<year>/<filter_state>/results/missing_plot_support_by_week_angle_cultivar_<year>.csv`
- `outputs/result_01_reflectance_distributions/<year>/ground_filtered/results/ground_filter_retention_<year>.csv`
- `outputs/result_01_reflectance_distributions/<year>/<filter_state>/results/vza_model_comparison_<year>.csv`
- `outputs/result_01_reflectance_distributions/<year>/<filter_state>/results/vza_model_terms_<year>.csv`
- `outputs/result_01_reflectance_distributions/<year>/<filter_state>/results/vza_top_interactions_<year>.csv`
- `outputs/result_01_raa_sun_geometry/<year>/<filter_state>/results/raa_vza_model_comparison_<year>.csv`
- `outputs/result_01_raa_sun_geometry/<year>/<filter_state>/results/raa_model_terms_<year>.csv`
- `outputs/result_01_raa_sun_geometry/<year>/<filter_state>/results/raa_top_interactions_<year>.csv`
- `outputs/result_01_reflectance_distributions/results/multiangular_evidence_summary.csv`
- `outputs/result_01_reflectance_distributions/results/angular_support_summary.csv`
- `outputs/result_01_reflectance_distributions/results/robustness_diagnostics.csv`
- `outputs/result_01_reflectance_distributions/results/vza_angular_support_summary.csv`
- `outputs/result_01_reflectance_distributions/results/raa_angular_support_summary.csv`
- `outputs/result_01_reflectance_distributions/results/2024_week_availability.csv`
- `outputs/result_01_reflectance_distributions/<year>/<filter_state>/figures/main/reflectance_distributions_by_vza.{pdf,png,svg,tiff}`
- `outputs/result_01_reflectance_distributions/<year>/<filter_state>/figures/main/reflectance_distributions_by_week_<year>.{pdf,png,svg,tiff}`
- `outputs/result_01_reflectance_distributions/<year>/<filter_state>/figures/main/angular_reflectance_curves.{pdf,png,svg,tiff}`
- `outputs/result_01_reflectance_distributions/<year>/<filter_state>/figures/main/angular_reflectance_curves_by_week.{pdf,png,svg,tiff}`
- `outputs/result_01_reflectance_distributions/<year>/<filter_state>/figures/main/angular_reflectance_curves_by_cultivar_<year>.{pdf,png,svg,tiff}`
- `outputs/result_01_reflectance_distributions/<year>/<filter_state>/figures/main/matched_off_nadir_effects.{pdf,png,svg,tiff}`
- `outputs/result_01_reflectance_distributions/<year>/<filter_state>/figures/main/seasonal_change_in_angular_contrast.{pdf,png,svg,tiff}`
- `outputs/result_01_reflectance_distributions/<year>/<filter_state>/figures/supplementary/reflectance_distributions_by_cultivar_week_<year>.{pdf,png,svg,tiff}`
- `outputs/result_01_reflectance_distributions/figures/main/angular_support_heatmap.{pdf,png,svg,tiff}`
- `outputs/result_01_reflectance_distributions/figures/main/observation_geometry_distribution.{pdf,png,svg,tiff}`
- `outputs/result_01_reflectance_distributions/figures/main/vza_contrast_2024_2025_comparison.{pdf,png,svg,tiff}`
- `outputs/result_01_reflectance_distributions/figures/main/multiangular_workflow_schematic.{pdf,png,svg,tiff}`
- `/run/media/davidem/Heim/2024/recovered_weeks/<week>/output/*.parquet`, for successfully extracted weeks
- `/run/media/davidem/Heim/2024/recovered_weeks/<week>/output/plots/*.parquet`, for successfully extracted weeks
- `/run/media/davidem/Heim/2025/recovered_weeks/<week>/output/*.parquet`, for successfully extracted 2025 weeks
- `/run/media/davidem/Heim/2025/recovered_weeks/<week>/output/plots/*.parquet`, for successfully extracted 2025 weeks
- `outputs/result_01_reflectance_distributions/<year>/<filter_state>/reports/reflectance_distributions_summary_<year>.md`
- `outputs/result_01_reflectance_distributions/<year>/<filter_state>/reports/vza_detailed_results_<year>.md`
- `outputs/result_01_reflectance_distributions/<year>/<filter_state>/reports/figure_captions_<year>.md`
- `outputs/result_01_reflectance_distributions/reports/multiangular_evidence_summary.md`
- `outputs/result_01_reflectance_distributions/reports/angular_support_summary.md`
- `outputs/result_01_reflectance_distributions/reports/robustness_diagnostics.md`
- `outputs/result_01_reflectance_distributions/reports/result_01_final_story.md`
- `Documentation/multiangular_reporting_literature.md`
- `outputs/result_01_reflectance_distributions/reports/paper_reporting_synthesis.md`
- `outputs/result_01_reflectance_distributions/manifests/external_data_manifest.csv`
- a timestamped profiling log in `outputs/result_01_reflectance_distributions/<year>/<filter_state>/logs/`

The Markdown results summary must include the key numerical tables, a short interpretation, exact output paths, input configuration, filtering rules, VZA boundaries, included and excluded weeks, recovery provenance, cultivar labels and sample sizes, model specification, and the number of plot-week observations used.

## Implementation Checklist

### Missing-Week Extraction

- [x] Create the self-contained `outputs/result_01_reflectance_distributions/` folder structure.
- [x] Add a Result 1 `README.md` describing purpose, commands, inputs, outputs, and external dependencies.
- [x] Confirm that temporary and large generated files are covered by repository `.gitignore` rules.
- [x] Confirm 2024 as the analysis season.
- [x] Inventory weeks 0--8 in both Heim locations without modifying the backup.
- [x] Record which weeks already have validated five-band per-plot parquets.
- [x] Check weeks 1, 2, 4, 6, and 7 for Metashape projects, camera geometry, DEMs, per-image orthophotos, and band structure.
- [x] Confirm that week 1 per-image orthophotos are three-band Byte/RGB products.
- [x] Confirm that week 6 has camera, DEM, orthomosaic, and five-band per-image orthophoto products in active Heim.
- [x] Create writable recovery directories under `/run/media/davidem/Heim/2024/recovered_weeks/`.
- [x] Create extraction/filtering configurations that read from the backup and write only to active Heim.
- [x] Create a week 6 configuration that reads its products from active Heim.
- [x] Extract weeks 2, 4, 6, and 7 through the existing module entry points without rerunning Metashape. Weeks 2, 4, 6, and 7 are complete and filtered to 24 plot parquets each. Week 7 was extracted from the read-only backup and wrote only to active Heim.
- [x] Validate new schemas, reflectance ranges, VZA ranges, plot IDs, image counts, and plot coverage. Weeks 2, 4, 6, and 7 passed image-level validation: week 2 has 242 image parquets, week 4 has 252, week 6 has 304, and week 7 has 259, with no corrupt files, no schema/range issues, and no files missing `plot_id`.
- [x] Produce `2024_week_availability.csv` with explicit inclusion/exclusion reasons.

### Data Preparation

- [x] Verify cultivar names and plot-to-cultivar assignments from the 2024 polygon metadata for the preliminary analysis.
- [x] Build a manifest of all included per-plot parquet files.
- [x] Profile every parquet read and summarize I/O timing for the preliminary feature-cache analysis.
- [x] Apply the documented vegetation and reflectance-quality filters for week 6 plot parquet generation.
- [x] Assign VZA classes using the fixed boundaries in this plan for the preliminary broad-bin analysis and add a 5-degree fine-bin mode from `10-15` through `50-55`; the sparse `55-60` bin is excluded for comparability.
- [x] Count plots and plot-week observations per VZA class for the preliminary analysis.
- [x] Flag sparse angle classes before plotting or modeling.
- [x] Aggregate recovered plot parquets into plot-week-angle-band features for weeks 0, 2, 3, 4, 5, 6, 7, and 8.
- [x] Verify that no plot-week-angle combination is duplicated in the preliminary long feature table.

### Descriptive Analysis

- [x] Calculate mean, median, standard deviation, IQR, and 5th/95th percentiles for validated weeks 0, 2, 3, 4, 5, 6, 7, and 8.
- [x] Calculate matched absolute and relative off-nadir contrasts, including paired Cohen's `dz`, for validated weeks 0, 2, 3, 4, 5, 6, 7, and 8.
- [x] Calculate matched temporal changes between available weeks.
- [x] Calculate changes in angular contrast between weeks.
- [x] Summarize reflectance curves separately by cultivar for all five bands in the preliminary figure.
- [x] Check whether missing plots differ systematically by week, angle, treatment, or cultivar.
- [x] Produce the same plot-level summaries, matched contrasts, statistics, and figures for validated 2025 weeks 0, 3, 5, and 7.
- [x] Apply a documented ground/background filter and regenerate the Result 1 statistics and figures for comparison with the unfiltered analysis. The implemented sensitivity uses `OSAVI > 0.2` before plot-level aggregation.
- [x] Derive relative azimuth angle from view azimuth and sun azimuth, then summarize reflectance by RAA class within matched VZA bins.

### Statistical Analysis

- [x] Fit a preliminary week-by-angle model using cluster-robust standard errors by plot.
- [x] Fit the cultivar-by-angle-by-week model when supported by sample size.
- [x] Fit reduced cultivar interaction models if the full model is unstable.
- [x] Report preliminary model estimates, effect sizes, 95% confidence intervals, and exact p-values.
- [ ] Inspect residuals, influential plots, heteroscedasticity, and convergence warnings.
- [ ] Run a sensitivity analysis with medians versus means.
- [ ] Check whether conclusions change after restricting to plots present in all included weeks.
- [x] Inspect VAA or relative-azimuth effects at matched VZA.
- [x] Test whether the ground-filtered analysis changes the sign, magnitude, or uncertainty of matched off-nadir contrasts.
- [x] Fit a supporting directional model or stratified comparison for RAA effects when sample support is sufficient.
- [x] Persist interpretable RAA model diagnostics, including R2, adjusted R2, Delta R2, Delta AIC/BIC, phase-angle coefficient, and reference-coded VZA x RAA coefficient tables.
- [x] Persist interpretable VZA model-ladder diagnostics, including controls-only, VZA, and VZA x week comparisons with R2, adjusted R2, Delta R2, Delta AIC/BIC, and reference-coded VZA x week coefficient tables.

### Figure Production

- [x] Implement the shared typography, panel labels, color palette, line styles, and export settings for preliminary figures.
- [x] Build Figure 1: seasonal angular distribution atlas.
- [x] Build Figure 2: angular reflectance fingerprints by week for validated weeks 0, 2, 3, 4, 5, 6, 7, and 8.
- [x] Build Figure 3: matched off-nadir effect forest plot for validated weeks 0, 2, 3, 4, 5, 6, 7, and 8.
- [x] Build Figure 4: cultivar angular response for all five bands for validated weeks 0, 2, 3, 4, 5, 6, 7, and 8.
- [x] Build Figure 5: seasonal change in angular contrast.
- [x] Build treatment-specific angular reflectance curves for all five bands and validated weeks.
- [x] Build cultivar-by-treatment angular reflectance curves to show whether cultivar patterns differ within treated and untreated plots.
- [ ] Produce supplementary individual-plot curves. Blue/green are now included in the all-band cultivar figure.
- [x] Produce 2025 versions of the main Result 1 figures using the same design, binning, and output naming pattern.
- [x] Produce ground-filtered sensitivity versions of the main Result 1 figures.
- [x] Produce RAA/sun-position diagnostic figures showing directional reflectance effects within VZA bins.
- [x] Produce final 2024/2025 VZA comparison figures with shared per-band y-axis limits.
- [x] Produce final angular support heatmaps for VZA and RAA-within-VZA support.
- [x] Produce final observation-geometry support curves.
- [x] Produce final workflow schematic separating nadir/orthomosaic and multiangular workflows.
- [ ] Add sample sizes and uncertainty to every inferential panel.
- [ ] Test figures at one-column and two-column journal widths.
- [ ] Check grayscale and color-vision accessibility.
- [x] Export preliminary PDF and 600 dpi PNG versions.
- [x] Export final SVG/TIFF submission versions.

### Reporting and Reproducibility

- [x] Keep all compact Result 1 artifacts inside the self-contained result folder.
- [x] Create `manifests/external_data_manifest.csv` for external Heim dependencies.
- [x] Verify that no raw image, raster, Metashape project, or large per-pixel parquet was copied into the repository.
- [ ] Check the final result-folder size and remove redundant or temporary outputs before committing.
- [x] Write the required timestamped profiling log for the preliminary analysis.
- [x] Save all preliminary numerical figure data as CSV or parquet.
- [x] Produce the Markdown result summary with a preliminary paper-ready results table.
- [x] Record input paths, recovery provenance, configuration, filtering rules, VZA classes, and excluded weeks for the preliminary analysis.
- [x] Record the random seed and bootstrap/model settings.
- [x] Draft captions using the caption checklist in this plan.
- [x] Create a literature-grounded reporting guide for multiangular reflectance, including local Rene Heim/ONCERCO context and external UAV/BRDF/disease-monitoring sources.
- [x] Create a paper-facing reporting synthesis that maps ONCERCO VZA/RAA result tables to manuscript-ready Methods, Results, Discussion, and limitation text.
- [x] Create a final paper-facing Result 1 evidence table combining VZA matched contrasts, RAA matched contrasts, Delta R2, Delta AIC/BIC, and plot support.
- [x] Create final angular-support and robustness-diagnostic tables for checking sampling balance, sparse RAA cells, model-row completeness, and mean-versus-median sensitivity.
- [x] Create `outputs/result_01_reflectance_distributions/reports/result_01_final_story.md` with main claim, evidence chain, paper figures, paper tables, methods text, results text, limitations, and reproducibility details.
- [ ] Add the final verified result to the LaTeX manuscript without overstating causality or significance.

## Current Execution Status

As of the latest local update, all processable missing 2024 weeks have been recovered into active Heim and validated. Week 2 produced 242 image-level parquets, week 4 produced 252 image-level parquets, week 6 produced 304 image-level parquets, and week 7 produced 259 image-level parquets. Each recovered week has 24 plot-level parquets. Week 7 plot filtering produced 170,737,613 rows across all 24 plots, with finite five-band reflectance, finite `OSAVI`, finite `ExcessGreen`, and VZA coverage from 0.00 to 50.86 degrees.

The Result 1 figures and report have also been regenerated from plot-level parquet files using 5-degree VZA bins from `10-15` through `50-55`. The sparse `55-60` bin was removed from all weeks so the panels use a common, adequately supported angle range. Week 7 has sparse support in `50-55` with only two matched plots, so that bin is retained in the full CSV but excluded from headline interpretation. The matched angular contrast table now reports median contrast, bootstrap confidence intervals, and paired Cohen's `dz` relative to the `10-15` reference class. The current strongest positive matched angular effects are late-season, especially in NIR and red edge during weeks 7 and 8, while week 6 adds a contrasting transitional pattern.

For 2025, the complete Metashape bundles identified during inventory were weeks 0, 3, 5, and 7. Weeks 0, 3, and 5 were re-extracted from active Heim into clean recovery folders instead of relying on the older plot outputs with nonfinite reflectance rows. Week 7 was recovered from the read-only backup and wrote only to active Heim. The clean 2025 recovery outputs are now:

| Week | Image parquets | Plot parquets | Plot rows | VZA range | Source |
|---:|---:|---:|---:|---|---|
| 0 | 255 | 24 | 162,868,376 | 0.00--57.40 | active Heim |
| 3 | 230 | 24 | 150,737,067 | 0.00--57.45 | active Heim |
| 5 | 241 | 24 | 163,902,375 | 0.00--53.28 | active Heim |
| 7 | 253 | 24 | 153,516,767 | 0.00--53.59 | read-only backup source, active Heim output |

All four recovered 2025 weeks passed image-level validation and plot-level checks for finite five-band reflectance, finite `OSAVI`, finite `ExcessGreen`, and complete 24-plot coverage. The provenance table is `outputs/result_01_reflectance_distributions/manifests/2025_recovery_manifest.csv`. Weeks 1, 2, 4, and 6 did not have complete Metashape bundles in the current inventory, so no additional 2025 week should be started unless new products are located.

The analysis script has been extended to run both years with:

```bash
python -m src.analysis.result_01_reflectance_distributions --year 2024 --fine-vza-bins
python -m src.analysis.result_01_reflectance_distributions --year 2025 --fine-vza-bins
python -m src.analysis.result_01_reflectance_distributions --year 2024 --fine-vza-bins --ground-filter --osavi-threshold 0.2
python -m src.analysis.result_01_reflectance_distributions --year 2025 --fine-vza-bins --ground-filter --osavi-threshold 0.2
```

New runs are organized under:

```text
outputs/result_01_reflectance_distributions/<year>/<filter_state>/
```

where `<filter_state>` is `unfiltered` or `ground_filtered`. The older root-level outputs remain as legacy artifacts and should not be treated as the current organized result package.

Current Result 1 generated tables include plot-week-angle features, reflectance-by-VZA summaries, matched angular contrasts, matched temporal changes, angular-contrast changes, cultivar angular comparisons, cultivar-angle model coefficients, missing plot-support diagnostics, and figure data for both years. The 2025 unfiltered analysis covers weeks 0, 3, 5, and 7 and produced:

| Output | Rows |
|---|---:|
| `reflectance_by_vza_summary.csv` | 180 |
| `matched_angular_contrasts.csv` | 160 |
| `temporal_reflectance_changes_2025.csv` | 135 |
| `angular_contrast_changes_2025.csv` | 120 |
| `cultivar_angular_comparison_2025.csv` | 360 |
| `cultivar_angle_model_2025.csv` | 365 |
| `missing_plot_support_by_week_angle_cultivar_2025.csv` | 144 |

The ground/background sensitivity analysis is complete for both years using `OSAVI > 0.2` applied before plot-level VZA aggregation. The filter was deliberately applied before summarizing reflectance so the ground-filtered figures are genuinely derived from filtered pixel samples rather than post-hoc filtered plot summaries.

| Year | Filter rule | Rows after basic VZA/band filter | Rows retained | Rows removed |
|---:|---|---:|---:|---:|
| 2024 | `OSAVI > 0.2` | 70,218,603 | 68,103,517 | 3.01% |
| 2025 | `OSAVI > 0.2` | 44,087,347 | 38,349,888 | 13.01% |

Ground-filtered outputs are stored at:

- `outputs/result_01_reflectance_distributions/2024/ground_filtered/`
- `outputs/result_01_reflectance_distributions/2025/ground_filtered/`

The ground-filtered reports explicitly document the threshold and row-retention accounting:

- `outputs/result_01_reflectance_distributions/2024/ground_filtered/reports/reflectance_distributions_summary_2024.md`
- `outputs/result_01_reflectance_distributions/2025/ground_filtered/reports/reflectance_distributions_summary_2025.md`

All main figures in the `ground_filtered/figures/` folders are filtered with `OSAVI > 0.2`. Figures outside `ground_filtered` should be interpreted as unfiltered unless their report states otherwise.

After inspecting the treatment-specific curves, the 2024 metadata join was corrected. The extraction pipeline assigns 2024 `plot_id` values from the GeoPackage row order when no explicit polygon ID field is present, while the first Result 1 implementation had joined 2024 metadata using `ifz_id - 90001`. Because the 2024 GeoPackage row order is reversed relative to increasing `ifz_id`, this swapped the treated and untreated labels in the generated 2024 Result 1 tables and figures. The metadata loader now uses the same row-order plot IDs as extraction:

- `plot_0` through `plot_11`: `trt`
- `plot_12` through `plot_23`: `no_trt`

The 2024 unfiltered and ground-filtered Result 1 tables, models, reports, and figures were regenerated from the existing aggregated feature parquets after this correction. A sanity check on week 8 NIR now matches the expected treatment direction:

| Filter state | Week | Treatment | Mean NIR | Median NIR | Plots |
|---|---:|---|---:|---:|---:|
| unfiltered | 8 | `trt` | 0.392168 | 0.396810 | 12 |
| unfiltered | 8 | `no_trt` | 0.208186 | 0.206553 | 12 |
| ground_filtered | 8 | `trt` | 0.392168 | 0.396810 | 12 |
| ground_filtered | 8 | `no_trt` | 0.208186 | 0.206553 | 12 |

Additional hidden-issue audit, 2026-06-16:

- The cached-feature loader now overwrites any cached `cult` and `trt` columns with the corrected polygon metadata, so stale cached 2024 labels cannot reintroduce the treatment swap.
- The fine-bin helper default OSAVI threshold now matches the CLI default: `0.2`.
- Unfiltered reruns now remove stale `ground_filter_retention_<year>.csv` files instead of leaving misleading retention tables in unfiltered output folders.
- The 2024 and 2025 unfiltered and ground-filtered Result 1 folders were regenerated from plot parquets with `--fine-vza-bins`; ground-filtered runs used `--ground-filter --osavi-threshold 0.2`.
- Post-regeneration validation confirmed that filtered feature tables differ from unfiltered tables:
  - 2024: 4,587 of 8,240 plot-week-band-angle rows changed; max absolute reflectance difference 0.038231; mean absolute difference 0.001276.
  - 2025: 2,760 of 4,115 plot-week-band-angle rows changed; max absolute reflectance difference 0.055616; mean absolute difference 0.004437.
- Retention tables are now present only under `ground_filtered/results/` and absent from `unfiltered/results/`.

RAA/sun-relative geometry support analysis, 2026-06-17:

- Implemented `python -m src.analysis.result_01_raa_sun_geometry` as a separate Result 1 support analysis.
- The analysis uses existing plot-parquet columns `vza`, `vaa`, `saa`, and `sunelev`; it does not recompute camera or sun geometry.
- Derived variables are `sza`, signed/absolute RAA, and phase angle. RAA classes are `0-45`, `45-90`, `90-135`, and `135-180`; VZA classes remain the current fine `10-15` through `50-55` bins.
- Generated unfiltered and `OSAVI > 0.2` ground-filtered outputs for both years under `outputs/result_01_raa_sun_geometry/<year>/<filter_state>/`.

| Year | Filter state | Plot-week-VZA-RAA feature rows | RAA summary rows | Matched RAA contrast rows | Model rows | Model term rows | Retention table |
|---:|---|---:|---:|---:|---:|---:|---|
| 2024 | unfiltered | 29,440 | 1,420 | 1,055 | 5 | 140 | no |
| 2024 | ground_filtered | 29,435 | 1,420 | 1,055 | 5 | 140 | yes |
| 2025 | unfiltered | 13,700 | 705 | 490 | 5 | 140 | no |
| 2025 | ground_filtered | 13,680 | 705 | 490 | 5 | 140 | yes |

The first RAA figure version was too large and unreadable, so the output figure design was revised. The folders now keep compact combined heatmap atlases, RAA curves, phase-angle curves, support heatmaps, and top matched-contrast figures; obsolete per-band heatmaps were removed to avoid confusion.

Interpretable model diagnostics were added to the RAA/sun-relative geometry output. The main model-comparison CSV now includes `vza_only_r2`, `vza_raa_r2`, `delta_r2_raa_vs_vza`, adjusted R2 values, Delta AIC/BIC, `phase_angle_estimate`, and `phase_angle_p`. The separate `raa_model_terms_<year>.csv` and `raa_top_interactions_<year>.csv` tables expose reference-coded RAA main effects and VZA x RAA interaction coefficients with standard errors, confidence intervals, and p-values. For paper interpretation, matched RAA contrasts remain the most direct effect-size table because they compare the same plot, week, band, and VZA bin against a common RAA reference.

VZA detailed reporting update, 2026-06-17:

- Added `python -m src.analysis.result_01_vza_interpretable_reports`.
- Regenerated detailed VZA reports for 2024 and 2025, both unfiltered and `OSAVI > 0.2` ground-filtered.
- Added model-ladder tables comparing `reflectance ~ week + cultivar + treatment`, `+ VZA class`, and `VZA class x week`.
- Added `vza_model_comparison_<year>.csv`, `vza_model_terms_<year>.csv`, and `vza_top_interactions_<year>.csv` under each Result 1 year/filter-state result folder.
- Added paper-ready reports at `outputs/result_01_reflectance_distributions/<year>/<filter_state>/reports/vza_detailed_results_<year>.md`.
- The detailed reports use matched VZA contrasts as the main interpretable effect-size evidence and require at least 10 matched plots for headline seasonal contrast-change tables.

Literature-grounded reporting update, 2026-06-17:

- Created `Documentation/multiangular_reporting_literature.md` to summarize multiangular/BRDF reporting conventions and map them to ONCERCO outputs.
- Created `outputs/result_01_reflectance_distributions/reports/paper_reporting_synthesis.md` with manuscript-ready claim language, numeric evidence, reporting tables, figure recommendations, and limitations.
- Inspected local read-only Rene Heim/ONCERCO context documents in the Heim backup. Relevant project documents include the DFG proposal, ON UAV progress report, BRDF presentation assets, and the leaf-dynamics CLS manuscript draft. Public web search did not locate a clear peer-reviewed Rene Heim multiangular reflectance paper, so any specific Rene Heim paper should be added manually if title/DOI/PDF is supplied.
