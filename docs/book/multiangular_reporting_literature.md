# Multiangular Reflectance Reporting Literature Notes

## Purpose

This note summarizes how multiangular vegetation reflectance studies generally report observation-geometry effects, and maps those conventions to the ONCERCO sugar beet VZA/RAA outputs.

The intended paper framing is:

> Multiangular UAV imagery contains structured reflectance variation linked to viewing and illumination geometry. This variation is especially visible in red edge and NIR, consistent with canopy-structure, leaf-angle, shadowing, and background-exposure effects. Result 1 demonstrates that the angular signal exists and is measurable; disease-prediction improvement should be tested in a later result.

## Local Rene Heim / ONCERCO Context

Local read-only archive documents were inspected under `/run/media/davidem/heim_data/Backup/proj_on_cerco/`.

Relevant local sources found:

| Source | Path | Relevant content | Use in manuscript |
|---|---|---|---|
| ONCERCO DFG project description | `/run/media/davidem/heim_data/Backup/proj_on_cerco/docs/20240130_oncerco_dfg_heim_public.pdf` | Frames the project as improving plant disease detection using oblique UAV observation angles from multispectral cameras. States that nadir-only acquisition is standard, but canopy reflectance depends on observation angle. Highlights sugar beet leaf inclination and lower-leaf disease development as motivation for oblique views. | Use as internal/project motivation, not as peer-reviewed evidence unless citation status is acceptable. |
| ON UAV progress report | `/run/media/davidem/heim_data/Backup/proj_on_cerco/docs/presentation/20231106_progressreport_onuav_heim.pdf` | Introduces BRDF geometry, observation-angle dependence of reflectance, sugar beet leaf angle distribution, and the empirical-vs-physical model motivation. Cites Nicodemus, Schaepman-Strub, Jay et al., and Dorigo. | Use to identify literature threads and confirm project rationale. |
| Leaf dynamics CLS draft | `/run/media/davidem/heim_data/Backup/proj_on_cerco/docs/manuscript/quaratiello_giuseppe/20240821_manuscript_draft_leafdynamics_cls_edRH-Rene’s MacBook Pro.docx` | Discusses LAI, LCC, LIA, canopy architecture, and CLS regime. Connects structural/biochemical traits with reflectance signatures. | Use as local biological context and to align discussion language around LIA/LAI/LCC. |
| BRDF presentation assets | `/run/media/davidem/heim_data/Backup/proj_on_cerco/docs/presentation/imgs/Brdf*.pdf` | BRDF explanatory figures. | Use only as internal design/context material unless permissions are clear. |

Web search for public Rene/René Heim multiangular papers did not return a clear peer-reviewed multiangular reflectance paper. If Rene has specific papers, add them manually once DOI/title/PDF is available.

## Literature Matrix

| Literature thread | Source | What papers commonly report | Relevance to ONCERCO |
|---|---|---|---|
| Reflectance geometry definitions | Nicodemus et al. 1977; Schaepman-Strub et al. 2006, reflectance quantities in optical remote sensing | Defines directional reflectance quantities and the need to specify illumination and viewing geometry. | Methods should define VZA, VAA, SZA, SAA, RAA, phase angle, and the reflectance factor being analyzed. |
| BRDF/NBAR operational precedent | NASA MODIS MCD43A4 NBAR product | NBAR products remove view-angle effects from directional reflectance to create stable, consistent nadir BRDF-adjusted reflectance. | Supports the claim that view-angle effects are large enough to require explicit handling in remote-sensing products. |
| Sentinel-2 NBAR/harmonization | Montero et al. 2024, `sen2nbar` | Motivates nadir BRDF adjustment for more comparable surface reflectance over time and geometry. | Supports the argument that uncorrected reflectance can vary with sun-view geometry and that geometry-aware reporting is standard. |
| UAV multiangular extraction | Schneider-Zapp et al. 2019 | Reports a UAV-compatible workflow for multiangular reflectance/HDRF from lightweight multispectral cameras and validates radiometry with RMSE. | Supports reporting acquisition geometry, calibration workflow, and fit/validation error when modelling reflectance factor. |
| UAV multiangular anisotropy visualization | Qin et al. 2026 | Extracts multiangular observations from UAV imagery and visualizes anisotropy in VZA/RAA space; reports strong red-edge/NIR anisotropy. | Directly supports ONCERCO VZA x RAA heatmaps/curves and band-specific anisotropy reporting. |
| Sugar beet multiangular biophysical retrieval | Jay et al. 2017, sugar beet multiangular optical remote sensing | Compares multiangular vegetation indices and PROSAIL inversion for LAI/chlorophyll/nitrogen retrieval. | Important crop-specific precedent that sugar beet canopy parameters are angle-sensitive. |
| Plant disease UAV motivation | Heim et al. project materials; Mahlein/Günder sugar beet/Cercospora work | UAV disease monitoring is useful, but most disease models use nadir-like imagery or plant-specific imagery rather than explicit multiangular geometry. | Justifies the gap: ONCERCO tests whether discarded off-nadir geometry may contain disease-relevant information. |
| Leaf angle and canopy structure | Quaratiello/Heim local draft and plant trait literature | LAI, LIA, LCC, canopy gaps, and leaf orientation drive reflectance and disease/stress response interpretation. | Discussion should interpret red edge/NIR angular effects as canopy-structure/leaf-angle effects, not only radiometric noise. |

## Reporting Pattern Used by Strong Multiangular Papers

### 1. Define observation geometry explicitly

Report:

- View zenith angle: `VZA`
- View azimuth angle: `VAA`
- Solar zenith angle: `SZA`
- Solar azimuth angle: `SAA`
- Relative azimuth angle: `RAA = abs(((SAA - VAA + 180) % 360) - 180)`
- Phase angle, when used
- VZA and RAA bin edges

ONCERCO implementation:

- VZA bins: `10-15` through `50-55`
- RAA bins: `0-45`, `45-90`, `90-135`, `135-180`
- Phase angle derived from `sunelev`, `vza`, and RAA

### 2. Report angular support before interpreting effects

Report:

- plots per VZA bin
- plots per RAA x VZA bin
- sparse-bin rule
- weeks and plot-week records
- missingness by cultivar/treatment where relevant

ONCERCO rule:

- Headline VZA/RAA effects require at least 10 matched plots.
- Sparse bins remain in full CSV outputs but are not headline claims.

### 3. Use matched effect sizes as the primary evidence

Preferred VZA effect:

```text
reflectance(plot, week, band, off-reference VZA)
- reflectance(plot, week, band, reference VZA)
```

Preferred RAA effect:

```text
reflectance(plot, week, band, VZA bin, RAA class)
- reflectance(plot, week, band, VZA bin, reference RAA class)
```

Report:

- median absolute contrast
- bootstrap 95% CI
- median relative contrast
- paired effect size, such as Cohen's `dz`
- matched plot count

### 4. Use models as support, not as the only evidence

Recommended VZA model ladder:

```text
controls: reflectance ~ week + cultivar + treatment
+ VZA:    reflectance ~ VZA class + week + cultivar + treatment
seasonal: reflectance ~ VZA class * week + cultivar + treatment
```

Recommended RAA support model:

```text
VZA-only: reflectance ~ VZA class + week + cultivar + treatment
RAA:      reflectance ~ VZA class + RAA class + VZA class:RAA class + week + cultivar + treatment
phase:    reflectance ~ VZA class + phase angle + week + cultivar + treatment
```

Report:

- R2 and adjusted R2
- Delta R2
- Delta AIC
- Delta BIC
- coefficient estimates with 95% CI
- exact model formulas

Interpretation rule:

- AIC can support added geometry terms.
- BIC is more conservative and penalizes categorical interaction models.
- Coefficients are reference-coded and should not replace matched contrasts for effect-size interpretation.

### 5. Interpret red edge and NIR carefully

General interpretation:

- NIR responds strongly to canopy structure, multiple scattering, shadows, canopy closure, and background exposure.
- Red edge is sensitive to canopy/pigment status and structural variation.
- Directional effects in these bands are biologically plausible for sugar beet because leaf angle and canopy architecture are central to the signal.

Avoid:

- claiming disease causality in Result 1;
- treating pixel counts as independent biological replicates;
- interpreting RPV failure as evidence that multiangular information is useless.

## ONCERCO Reporting Position

The most defensible statement is:

> The ONCERCO data show that sugar beet canopy reflectance is strongly observation-geometry dependent. VZA produces large matched reflectance changes, especially in NIR and red edge, and the VZA response changes over the season. RAA and phase-angle analyses show that VZA alone does not fully describe directional reflectance, indicating sun-view geometry and canopy anisotropy effects. These findings justify testing multiangular predictors against nadir-only predictors for disease prediction in subsequent analyses.

Do not state yet:

> Multiangular imagery improves disease prediction.

That must be supported by cross-validated disease models comparing nadir-only and multiangular features with year/plot-aware validation.

## Key Source Links

- Qin et al. 2026, UAV multiangular reflectance anisotropy: https://arxiv.org/abs/2606.10350
- Schneider-Zapp et al. 2019, UAV multiangular HDRF workflow: https://arxiv.org/abs/1905.03301
- NASA MODIS MCD43A4 NBAR product: https://lpdaac.usgs.gov/products/mcd43a4v061/
- NASA MODIS MCD43A1 BRDF/albedo model parameters: https://lpdaac.usgs.gov/products/mcd43a1v061/
- Montero et al. 2024, Sentinel-2 NBAR simplification: https://arxiv.org/abs/2404.15812
- Günder et al. 2022, UAV sugar beet/Cercospora data framework: https://arxiv.org/abs/2201.02885
- Günder et al. 2023, SugarViT disease severity prediction: https://arxiv.org/abs/2311.03076
