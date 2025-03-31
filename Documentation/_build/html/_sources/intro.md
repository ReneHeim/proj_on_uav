# Multi-angular UAV Reflectance Extractor

## Why is this repository important to you?

This repository contains code, written in [R](https://www.r-project.org/) and [Python](https://www.python.org/), to reproduce [LINK TO ARTCLE HERE]. If you are using any of the contained code or data, please use the following reference:

Heim, R. HJ., Okole, N., Steppe, K., van Labeke, M.C., Geedicke, I., & Maes, W. H. (2024). An applied framework to unlocking multi‑angular UAV reflectance data: A case study for classification of plant parameters in maize (*Zea mays*). *Precision Agriculture*. (accepted)

![alt text](https://github.com/ReneHeim/proj_on_uav/blob/main/graphical_abstract.png)

## What does this repository contain?

- The [ref](https://github.com/ReneHeim/proj_on_uav/tree/main/ref) directory containing the code that was used to clean and structure the data that was collected for the study XYZ, published here (LINK TBA).
- The [main](https://github.com/ReneHeim/proj_on_uav/tree/main/main) directory containing the code to re-run the presented method as it was done in the published manuscript.
- The [main_public](https://github.com/ReneHeim/proj_on_uav/tree/main/main_public) directory containing the code to run our method as it is intended for a new user.
- The [analysis](https://github.com/ReneHeim/proj_on_uav/tree/main/analysis) directory containing the code that was used to generate the results and visualizations as they were published HERE (LINK TBA).

## How to use this method to unlock multi-angular reflectance data?

### Installing required software

**Python:** Installing Python 3.X through the [Anaconda distribution](https://professorkazarinoff.github.io/Problem-Solving-101-with-Python/01-What-is-Python/01.03-What-is-Anaconda/). Please follow the instructions, based on your operating system, [HERE](https://docs.anaconda.com/anaconda/install/index.html).

**Agisoft Metashape:** Please download [Agisoft Metashape Professional](https://www.agisoft.com/downloads/installer/) and purchase a license to allow full functionality.

**Exiftool (by Phil Harvey):** Please follow the [instructions](https://exiftool.org/install.html) to install Exiftool on your operating system and associate the programm with your Python environment.

**PyExifTool and other Python libraries:** Please use THIS requirements file and the following command to install all necessary Python libraries and your preferred conda environment simultaniously:

`conda env create --name my-env-name --file environment.yml`

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
2. Open the [config_file.yaml](https://github.com/ReneHeim/proj_on_uav/blob/main/main_public/config_file.yaml)
3. Change the paths, settings, and output according to your specific setup
4. Execute the [01_main_extract_vza_aza_reflectance.py](https://github.com/ReneHeim/proj_on_uav/blob/main/main_public/01_main_extract_vza_aza_reflectance.py) with you IDE of choice (the results will be stored in the directory that was specified under *output* in step 3)
5. Execute the [02_filter_sample_location.py](https://github.com/ReneHeim/proj_on_uav/blob/main/main_public/02_filter_sample_location.py) with you IDE of choice (the results will be stored in the directory that was specified under *output* in step 3)
6. Execute the [03_merging_sample_locations.py](https://github.com/ReneHeim/proj_on_uav/blob/main/main_public/03_merging_sample_locations.py) with you IDE of choice (the results will be stored in the directory that was specified under *output* in step 3)
7. Execute the [04_orthomosaic_pixel_data.py](https://github.com/ReneHeim/proj_on_uav/blob/main/main_public/04_orthomosaic_pixel_data.py) with you IDE of choice (the results will be stored in the directory that was specified under *output* in step 3)

### Contact

If you have any questions how to use the code, please commit an issue for others to benefit from it. If this is not an option for you, please contact either Nathan Okole (okole@ifz-goettingen.de) or Rene Heim (rheim@uni-bonn.de)


```{tableofcontents}
```
