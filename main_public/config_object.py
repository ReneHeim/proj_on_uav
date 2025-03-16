# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 23:29:03 2023

@author: hp
"""

import yaml
from metadict import MetaDict
import os

input_config_file = r"config_file.yaml"

with open(input_config_file, "r") as file:
    input_config = yaml.load(file,  yaml.Loader)

input_config = MetaDict(input_config)

class config_object():
    def __init__(self, input_config):
        self.main_extract_out = os.path.join(input_config.outputs.paths.main_out, "extract")
        self.main_extract_cam_path = input_config.inputs.paths.cam_path
        self.main_extract_dem_path = input_config.inputs.paths.dem_path
        self.main_extract_ori = input_config.inputs.paths.ori
        self.main_extract_name = input_config.inputs.settings.file_name
        self.main_extract_path_list_tag = input_config.inputs.paths.orthophoto_path
        self.main_polygon_path = input_config.inputs.paths.polygon_file_path

        
        self.filter_input_dir = self.main_extract_out
        self.filter_out = os.path.join(input_config.outputs.paths.main_out, "filter")
        self.filter_radius = input_config.inputs.settings.filter_radius
        self.filter_groung_truth_coordinates = input_config.inputs.paths.ground_truth_coordinates
        
        self.merging_input_dir = self.filter_out
        self.merging_out = os.path.join(input_config.outputs.paths.main_out, "merge")
        self.merging_groung_truth_coordinates = self.filter_groung_truth_coordinates
        
        self.orthomosaic_ortho_path = input_config.inputs.paths.mosaic_path
        self.orthomosaic_dem_path = self.main_extract_dem_path
        self.orthomosaic_name = self.main_extract_name + "_for_classification_mosaic"
        self.orthomosaic_out = self.merging_out
        self.orthomosaic_radius = self.filter_radius
        self.precision = input_config.inputs.settings.precision  # default to 0.01 or any suitable value


config = config_object(input_config)