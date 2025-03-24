from rtree import index

import geopandas as gpd

from shapely.geometry import box

import rasterio

import os



# Load polygons (e.g., from a shapefile or GeoJSON)

polygons_gdf = gpd.read_file("your_polygons_file.shp")



# Create an R-tree spatial index for the polygons

spatial_index = index.Index()



# Map polygon IDs to their geometries

poly_geometries = {}

for idx, polygon in enumerate(polygons_gdf.geometry):

    spatial_index.insert(idx, polygon.bounds)  # Insert the polygon's bounding box into the index

    poly_geometries[idx] = polygon



# Folder containing the raster images

raster_folder = "path_to_raster_images"



# List to store intersecting rasters

intersecting_rasters = []



# Iterate over raster files and check intersections

for raster_file in os.listdir(raster_folder):

    if raster_file.endswith(('.tif', '.img')):

        raster_path = os.path.join(raster_folder, raster_file)



        with rasterio.open(raster_path) as src:

            # Get the raster bounds as a Shapely box

            raster_bounds = box(*src.bounds)



            # Use the spatial index to get candidate polygons

            candidate_polygons = list(spatial_index.intersection(raster_bounds.bounds))



            # Check precise intersection with candidate polygons

            for idx in candidate_polygons:

                if raster_bounds.intersects(poly_geometries[idx]):

                    intersecting_rasters.append(raster_file)

                    break  # No need to check other polygons for this raster



# Output intersecting rasters

print("Intersecting raster files:")

print(intersecting_rasters)