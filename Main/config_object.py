import os
import glob
import warnings
import yaml
from metadict import MetaDict


class config_object():
    def __init__(self, config_file_path):
        self.config_file_path = config_file_path

        # Load and resolve configuration
        self.input_config = self.load_and_prepare_config()
        self.setup_attributes()
        self.validate_all_paths()

    def load_and_prepare_config(self):
        with open(self.config_file_path, "r") as file:
            raw_config = yaml.load(file, yaml.Loader)

        base = raw_config.get('base_path', '')
        for section in ['inputs', 'outputs']:
            if section in raw_config:
                for key, val in raw_config[section].get('paths', {}).items():
                    if isinstance(val, str):
                        raw_config[section]['paths'][key] = val.replace("{base_path}", base)
                    elif isinstance(val, list):
                        raw_config[section]['paths'][key] = [v.replace("{base_path}", base) for v in val]

        return MetaDict(raw_config)

    def setup_attributes(self):
        cfg = self.input_config

        self.start_date = cfg.inputs.date_time.start
        self.time_zone = cfg.inputs.date_time.time_zone

        self.main_extract_out = os.path.join(cfg.outputs.paths.main_out, "extract")
        self.main_extract_cam_path = cfg.inputs.paths.cam_path
        self.main_extract_dem_path = cfg.inputs.paths.dem_path
        self.main_extract_ori = cfg.inputs.paths.ori
        self.main_extract_name = cfg.inputs.settings.file_name
        self.main_extract_path_list_tag = cfg.inputs.paths.orthophoto_path
        self.main_polygon_path = cfg.inputs.paths.get('polygon_file_path', None)

        self.filter_input_dir = self.main_extract_out
        self.filter_out = os.path.join(cfg.outputs.paths.main_out, "filter")
        self.filter_radius = cfg.inputs.settings.filter_radius
        self.filter_groung_truth_coordinates = cfg.inputs.paths.ground_truth_coordinates

        self.merging_input_dir = self.filter_out
        self.merging_out = os.path.join(cfg.outputs.paths.main_out, "merge")
        self.merging_groung_truth_coordinates = self.filter_groung_truth_coordinates

        self.orthomosaic_ortho_path = cfg.inputs.paths.mosaic_path
        self.orthomosaic_dem_path = self.main_extract_dem_path
        self.orthomosaic_name = self.main_extract_name + "_for_classification_mosaic"
        self.orthomosaic_out = self.merging_out
        self.orthomosaic_radius = self.filter_radius
        self.precision = cfg.inputs.settings.precision

    def _validate_path(self, path, path_type='file', allow_glob=False):
        if allow_glob:
            matched_paths = glob.glob(path)
            if not matched_paths:
                warnings.warn(f"[Path Warning] No files matched the glob pattern: {path}")
            return

        if not os.path.exists(path):
            warnings.warn(f"[Path Warning] Path does not exist: {path}")
            return

        if path_type == 'file' and not os.path.isfile(path):
            warnings.warn(f"[Path Warning] Expected a file, but got something else: {path}")
        elif path_type == 'dir' and not os.path.isdir(path):
            warnings.warn(f"[Path Warning] Expected a directory, but got something else: {path}")

    def validate_all_paths(self):
        self._validate_path(self.main_extract_cam_path, 'file')
        self._validate_path(self.main_extract_dem_path, 'file')
        self._validate_path(self.filter_groung_truth_coordinates, 'file')
        self._validate_path(self.orthomosaic_ortho_path, 'file')
        self._validate_path(self.main_extract_path_list_tag, allow_glob=True)

        for ori_path in self.main_extract_ori:
            self._validate_path(ori_path, 'dir')

        self._validate_path(self.filter_input_dir, 'dir')
        self._validate_path(self.merging_input_dir, 'dir')

        if self.main_polygon_path:
            self._validate_path(self.main_polygon_path, 'file')

        print("Path validation completed (warnings issued where needed).")
