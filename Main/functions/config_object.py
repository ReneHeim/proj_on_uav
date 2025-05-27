from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
import yaml, glob, warnings, logging
from metadict import MetaDict

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DateCfg:
    start: str
    tz: str


@dataclass(frozen=True)
class PathGroup:
    cam: Path
    dem: Path
    ori: list[Path]
    orthos: str
    polygon: Path | None = None

@dataclass(frozen=True)
class Settings:
    file_name: str
    filter_radius: float
    bands: list[int]
    target_crs: str


def _load_yaml(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh)

def _resolve_base(cfg: dict) -> MetaDict:
    base = cfg.get("base_path", "")
    for sect in ("inputs", "outputs"):
        for k, v in cfg.get(sect, {}).get("paths", {}).items():
            repl = lambda s: s.replace("{base_path}", base)
            cfg[sect]["paths"][k] = [repl(x) for x in v] if isinstance(v, list) else repl(v)
    return MetaDict(cfg)

def _check(path: str | Path, *, expect="file", allow_glob=False) -> None:
    path = str(path)
    if allow_glob:
        if not glob.glob(path):
            warnings.warn(f"[Path] glob empty → {path}")
        return
    if not Path(path).exists():
        warnings.warn(f"[Path] missing → {path}")
    elif expect == "file" and not Path(path).is_file():
        warnings.warn(f"[Path] expected file → {path}")
    elif expect == "dir" and not Path(path).is_dir():
        warnings.warn(f"[Path] expected dir → {path}")

class Config:
    """
    Read YAML, expand {base_path}, expose attributes, validate paths.
    Existing code that accessed attributes like `main_extract_out` still works.
    """

    # ── constructor ──────────────────────────────────────────────
    def __init__(self, cfg_path: str | Path) -> None:
        self._raw  = _load_yaml(Path(cfg_path))
        self._cfg  = _resolve_base(self._raw)
        self._set_attrs()
        self._validate()

    # ── attribute wiring ─────────────────────
    def _set_attrs(self) -> None:
        inp, out = self._cfg.inputs, self._cfg.outputs
        settings = inp.settings

        dt = DateCfg(inp.date_time.start, inp.date_time.time_zone)
        self.start_date, self.time_zone = dt.start, dt.tz

        paths = PathGroup(
            cam     = Path(inp.paths.cam_path),
            dem     = Path(inp.paths.dem_path),
            ori     = [Path(p) for p in inp.paths.ori],
            orthos  = inp.paths.orthophoto_path,
            polygon = inp.paths.get("polygon_file_path")
        )
        #legacy field names -------------------------------------------------
        self.main_extract_cam_path  = paths.cam
        self.main_extract_dem_path  = paths.dem
        self.main_extract_ori       = paths.ori
        self.main_extract_path_list_tag = paths.orthos
        self.main_polygon_path      = paths.polygon

        base_out = Path(out.paths.main_out)
        self.main_extract_out = base_out / "extract"
        self.filter_out       = base_out / "filter"
        self.merging_out      = base_out / "merge"
        self.plot_out         = base_out / "plots"
        self.main_extract_out_polygons_df = self.main_extract_out / "polygon_df"

        self.filter_groung_truth_coordinates = inp.paths.ground_truth_coordinates
        self.main_extract_name = settings.file_name
        self.filter_radius  = inp.settings.filter_radius
        self.bands          = inp.settings.bands
        self.target_crs     = inp.settings.target_crs

    # ── validation & mkdirs ─────────
    def _validate(self) -> None:
        inp = self._cfg.inputs
        # files to exist
        for f in [
            self.main_extract_cam_path,
            self.main_extract_dem_path,
            self.filter_groung_truth_coordinates,
            inp.paths.mosaic_path         # example
        ]:
            _check(f, expect="file")

        # glob pattern
        _check(self.main_extract_path_list_tag, allow_glob=True)

        # ensure needed dirs
        for d in [
            self.main_extract_out,
            self.filter_out,
            self.merging_out,
            self.plot_out,
            self.plot_out / "bands_data",
            self.plot_out / "merge_data",
            self.plot_out / "polygon_filtering_data",
            self.plot_out / "angles_data" / "top_down",
            self.plot_out / "angles_data" / "side_view",
            self.main_extract_out / "polygon_df",
        ]:
            Path(d).mkdir(parents=True, exist_ok=True)

        log.info("Path validation completed (warnings issued if needed).")

config_object = Config