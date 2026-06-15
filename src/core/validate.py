"""
Post-extraction data validation.

Run after pipeline_extract_data.py to verify:
  - Schema consistency across all parquets
  - Required columns present
  - Value ranges within bounds
  - No corrupt/unreadable files

Usage:
    python -m src.core.validate --dir /path/to/extract/output

    # From within the extraction pipeline:
    from src.core.validate import validate_extract_output
    issues = validate_extract_output(Path(config.main_extract_out))
"""

import logging
from pathlib import Path
from typing import Dict, List, Set

import polars as pl

REQUIRED_COLUMNS = [
    "Xw",
    "Yw",
    "band1",
    "band2",
    "band3",
    "band4",
    "band5",
    "elev",
    "delta_z",
    "delta_x",
    "delta_y",
    "distance_xy",
    "angle_rad",
    "vza",
    "vaa_rad",
    "vaa_temp",
    "vaa",
    "xcam",
    "ycam",
    "sunelev",
    "saa",
    "path",
]

OPTIONAL_COLUMNS = ["plot_id", "OSAVI", "ExcessGreen"]

QUALITY_CHECKS = {
    "vza": {"min": 0.0, "max": 90.0, "label": "view zenith angle"},
    "vaa": {"min": 0.0, "max": 360.0, "label": "view azimuth angle"},
    "sunelev": {"min": -10.0, "max": 90.0, "label": "sun elevation"},
    "saa": {"min": 0.0, "max": 360.0, "label": "solar azimuth"},
    "elev": {"min": -500.0, "max": 9000.0, "label": "elevation"},
}

BAND_RANGE = {"min": 0.0, "max": 2.0, "label": "band reflectance"}


def validate_extract_output(output_dir: Path, *, raise_on_error: bool = False) -> Dict:
    """
    Validate all .parquet files in an extraction output directory.

    Returns dict with:
        ok: bool
        n_files: int
        total_rows: int
        n_corrupt: int
        schema_issues: list of dicts
        range_issues: list of str
        missing_columns: list
        extra_columns: list
        files_without_plot_id: list of str
    """
    result = {
        "ok": True,
        "n_files": 0,
        "total_rows": 0,
        "n_corrupt": 0,
        "schema_issues": [],
        "range_issues": [],
        "missing_columns": [],
        "extra_columns": [],
        "files_without_plot_id": [],
    }

    files = sorted(output_dir.glob("*.parquet"))
    if not files:
        result["ok"] = False
        logging.error(f"No parquet files found in {output_dir}")
        return result

    result["n_files"] = len(files)

    # --- Phase 1: Read schemas, find reference ---
    schemas: Dict[str, Set] = {}
    reference_cols: Set[str] | None = None
    corrupt = []

    for f in files:
        try:
            fschema = pl.read_parquet_schema(f)
            schemas[f.name] = set(fschema.keys())
            if reference_cols is None:
                reference_cols = schemas[f.name]
        except Exception:
            corrupt.append(f.name)
            schemas[f.name] = set()

    result["n_corrupt"] = len(corrupt)
    if corrupt:
        result["ok"] = False
        for c in corrupt:
            logging.error(f"Corrupt parquet: {c}")
            result["schema_issues"].append({"file": c, "issue": "corrupt/unreadable"})

    if reference_cols is None:
        result["ok"] = False
        return result

    # --- Phase 2: Check schemas ---
    missing_req = [c for c in REQUIRED_COLUMNS if c not in reference_cols]
    result["missing_columns"] = missing_req
    if missing_req:
        result["ok"] = False
        logging.error(f"Reference schema missing required: {missing_req}")

    extra_cols = [
        c for c in reference_cols if c not in REQUIRED_COLUMNS and c not in OPTIONAL_COLUMNS
    ]
    result["extra_columns"] = extra_cols

    without_plot = [f for f, s in schemas.items() if "plot_id" not in s]
    result["files_without_plot_id"] = without_plot
    if without_plot:
        logging.warning(f"{len(without_plot)} files lack plot_id (--no-polygon was likely used)")

    inconsistent = [f for f, s in schemas.items() if s != reference_cols and f not in corrupt]
    if inconsistent:
        result["ok"] = False
        logging.error(f"{len(inconsistent)} files have inconsistent schema vs reference")
        for inc in inconsistent[:5]:
            extra = schemas[inc] - reference_cols
            missing = reference_cols - schemas[inc]
            detail = []
            if extra:
                detail.append(f"+{extra}")
            if missing:
                detail.append(f"-{missing}")
            result["schema_issues"].append({"file": inc, "issue": ", ".join(detail)})

    # --- Phase 3: Value range checks (sample) ---
    sample_files = [f for f in files if f.name not in corrupt][:3]
    for f in sample_files:
        try:
            df = pl.read_parquet(f)
            result["total_rows"] += df.height
        except Exception:
            continue

        for col, bounds in QUALITY_CHECKS.items():
            if col not in df.columns:
                continue
            vals = df[col].drop_nulls()
            if vals.len() == 0:
                result["range_issues"].append(f"{f.name}: {col} all null")
                result["ok"] = False
                continue

            vmin, vmax = vals.min(), vals.max()
            if vmin < bounds["min"]:
                msg = f"{f.name}: {col} min={vmin:.1f} < {bounds['min']} ({bounds['label']})"
                result["range_issues"].append(msg)
                result["ok"] = False
            if vmax > bounds["max"]:
                msg = f"{f.name}: {col} max={vmax:.1f} > {bounds['max']} ({bounds['label']})"
                result["range_issues"].append(msg)
                result["ok"] = False

        for bcol in [c for c in df.columns if c.startswith("band")]:
            vals = df[bcol].drop_nulls()
            vals = vals.filter(vals.is_finite())
            if vals.len() == 0:
                result["range_issues"].append(f"{f.name}: {bcol} has no finite values")
                result["ok"] = False
                continue
            vmin, vmax = vals.min(), vals.max()
            if vmax > BAND_RANGE["max"] * 2:
                result["range_issues"].append(
                    f"{f.name}: {bcol} max={vmax:.2f} >> {BAND_RANGE['max']}"
                )
                result["ok"] = False
            if vmin < -0.01 and bcol != "band_label":
                result["range_issues"].append(f"{f.name}: {bcol} min={vmin:.4f} < 0")
                result["ok"] = False

    # --- Phase 4: Summary ---
    if result["ok"]:
        logging.info(
            f"Validation PASSED: {result['n_files']} files, "
            f"{result['total_rows']:,} rows sampled, "
            f"{result['n_corrupt']} corrupt"
        )
    else:
        logging.error(
            f"Validation FAILED: {result['n_files']} files, "
            f"{len(result['schema_issues'])} schema issues, "
            f"{len(result['range_issues'])} range issues, "
            f"{result['n_corrupt']} corrupt, "
            f"{len(result['files_without_plot_id'])} missing plot_id"
        )

    if raise_on_error and not result["ok"]:
        raise ValueError(f"Data validation failed: see log for details")

    return result


def validate_single_parquet(path: Path) -> Dict:
    """Validate a single parquet file. Returns same structure as validate_extract_output."""
    import tempfile

    tmpdir = Path(tempfile.mkdtemp())
    import shutil

    dst = tmpdir / path.name
    shutil.copy2(path, dst)
    result = validate_extract_output(tmpdir)
    shutil.rmtree(tmpdir)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate extraction output")
    parser.add_argument("--dir", type=Path, required=True, help="Extract output directory")
    parser.add_argument("--raise", dest="raise_on_error", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = validate_extract_output(Path(args.dir), raise_on_error=args.raise_on_error)

    import json

    print(json.dumps(result, indent=2, default=str))
