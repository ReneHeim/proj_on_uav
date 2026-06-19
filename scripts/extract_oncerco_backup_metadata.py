#!/usr/bin/env python3
"""Extract and summarize ONCERCO backup spreadsheets into local CSV outputs."""

from __future__ import annotations

import argparse
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path("/run/media/davidem/heim_data/Backup/proj_on_cerco")
OUTPUT_ROOT = ROOT / "outputs/backup_metadata"
CSV_ROOT = OUTPUT_ROOT / "csv"
REPORT_ROOT = OUTPUT_ROOT / "reports"
MANIFEST_ROOT = OUTPUT_ROOT / "manifests"
LOG_ROOT = ROOT / "outputs/logs"

INCLUDE_KEYWORDS = (
    "dsdi",
    "lia",
    "lai",
    "lcc",
    "mta",
    "checklist",
    "versuchsplan",
    "analysis_checklist",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def configure_logging() -> Path:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    path = LOG_ROOT / f"extract_oncerco_backup_metadata_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(path), logging.StreamHandler()],
        force=True,
    )
    return path


def slugify(value: str) -> str:
    value = value.strip().replace(" ", "_")
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_.")


def classify(rel_path: Path) -> str:
    lower = str(rel_path).lower()
    if "dsdi" in lower:
        return "DSDI"
    if "lia" in lower:
        return "LIA"
    if "lai" in lower:
        return "LAI"
    if "lcc" in lower:
        return "LCC"
    if "mta" in lower:
        return "MTA"
    if "checklist" in lower:
        return "CHECKLIST"
    if "versuchsplan" in lower:
        return "TRIAL_PLAN"
    return "OTHER"


def should_include(path: Path) -> bool:
    lower = str(path).lower()
    return path.suffix.lower() in {".xlsx", ".xls"} and any(
        keyword in lower for keyword in INCLUDE_KEYWORDS
    )


def read_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name=sheet_name)
    frame.columns = [str(col).strip() for col in frame.columns]
    return frame


def summarize_numeric(frame: pd.DataFrame) -> dict[str, float | int | None]:
    numeric = frame.select_dtypes(include="number")
    if numeric.empty:
        return {
            "numeric_columns": 0,
            "numeric_mean": None,
            "numeric_min": None,
            "numeric_max": None,
        }
    return {
        "numeric_columns": int(numeric.shape[1]),
        "numeric_mean": float(numeric.stack().mean()),
        "numeric_min": float(numeric.min().min()),
        "numeric_max": float(numeric.max().max()),
    }


def summarize_measurements(kind: str, frame: pd.DataFrame) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {}
    if kind == "DSDI":
        ds_plot = frame["ds_plot"] if "ds_plot" in frame.columns else None
        if ds_plot is not None:
            result["ds_plot_mean"] = float(pd.to_numeric(ds_plot, errors="coerce").mean())
        leaf_cols = [c for c in frame.columns if c.startswith("ds_leaf")]
        if leaf_cols:
            result["ds_leaf_mean"] = float(
                pd.to_numeric(frame[leaf_cols].stack(), errors="coerce").mean()
            )
        di_cols = [c for c in frame.columns if c.startswith("di_leaf")]
        if di_cols:
            result["di_leaf_mean"] = float(
                pd.to_numeric(frame[di_cols].stack(), errors="coerce").mean()
            )
    elif kind == "LIA":
        lia_cols = [c for c in frame.columns if c.upper().startswith("LIA(")]
        if lia_cols:
            result["lia_mean"] = float(
                pd.to_numeric(frame[lia_cols].stack(), errors="coerce").mean()
            )
            result["lia_min"] = float(pd.to_numeric(frame[lia_cols].stack(), errors="coerce").min())
            result["lia_max"] = float(pd.to_numeric(frame[lia_cols].stack(), errors="coerce").max())
    elif kind == "LAI":
        for col in ["LAI", "LAIc"]:
            if col in frame.columns:
                result[f"{col.lower()}_mean"] = float(
                    pd.to_numeric(frame[col], errors="coerce").mean()
                )
    elif kind == "LCC":
        for col in ["LCC", "LCC_mean"]:
            if col in frame.columns:
                result[f"{col.lower()}_mean"] = float(
                    pd.to_numeric(frame[col], errors="coerce").mean()
                )
    return result


def export_workbook(path: Path, source_root: Path, output_root: Path) -> list[dict[str, object]]:
    started = time.perf_counter()
    rel_path = path.relative_to(source_root)
    kind = classify(rel_path)
    workbook_out = output_root / "csv" / rel_path.with_suffix("")
    workbook_out.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    xls = pd.ExcelFile(path)
    for sheet_name in xls.sheet_names:
        sheet_started = time.perf_counter()
        frame = read_sheet(path, sheet_name)
        if frame.empty:
            logging.info("[SHEET] %s :: %s skipped (empty)", rel_path, sheet_name)
            continue
        sheet_slug = slugify(sheet_name)
        csv_path = workbook_out / f"{sheet_slug}.csv"
        frame.to_csv(csv_path, index=False)
        numeric_summary = summarize_numeric(frame)
        measurement_summary = summarize_measurements(kind, frame)
        logging.info(
            "[SHEET] %s :: %s rows=%d cols=%d exported=%s in %.2fs",
            rel_path,
            sheet_name,
            frame.shape[0],
            frame.shape[1],
            csv_path.relative_to(output_root),
            time.perf_counter() - sheet_started,
        )
        records.append(
            {
                "workbook": str(rel_path),
                "sheet": sheet_name,
                "kind": kind,
                "rows": int(frame.shape[0]),
                "cols": int(frame.shape[1]),
                "csv_path": str(csv_path.relative_to(output_root)),
                "columns_preview": ", ".join(frame.columns[:10]),
                **numeric_summary,
                **measurement_summary,
            }
        )
    logging.info("[WORKBOOK] %s exported in %.2fs", rel_path, time.perf_counter() - started)
    return records


def write_report(output_root: Path, inventory: pd.DataFrame, log_path: Path) -> None:
    report_path = output_root / "reports/backup_metadata_summary.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ONCERCO Backup Metadata Summary",
        "",
        "This package extracts the spreadsheet-based trial metadata, disease scores, and canopy measurements from the read-only ONCERCO backup into CSV files under `outputs/backup_metadata/`.",
        "",
        "## Interpretation",
        "",
        "The backup spreadsheets confirm that ONCERCO is not only an imagery project. The archive also contains plot-level disease scoring (`DSDI`), leaf inclination angle (`LIA`), leaf area index (`LAI`), leaf chlorophyll content (`LCC`), multi-temporal architecture measurements (`MTA`), and proposal/checklist files that define the trial design and data-collection schedule.",
        "",
        "The strongest complementary measurements for the reflectance work are `DSDI`, `LIA`, `LAI`, and `LCC` because they can be joined to plot IDs, cultivars, treatments, and weeks. Those variables can explain whether angular reflectance differences are driven by canopy architecture, physiology, or disease progression.",
        "",
        "## Key Tables",
        "",
        "| Workbook | Sheet | Kind | Rows | Cols | Notes |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in inventory.head(25).to_dict(orient="records"):
        notes = row.get("columns_preview", "")
        lines.append(
            f"| `{row['workbook']}` | `{row['sheet']}` | {row['kind']} | {row['rows']} | {row['cols']} | {notes} |"
        )
    lines += [
        "",
        "## Files",
        "",
        f"- Inventory CSV: `{output_root / 'manifests/backup_metadata_inventory.csv'}`",
        f"- Extracted CSV root: `{output_root / 'csv'}`",
        f"- Log: `{log_path}`",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_readme(output_root: Path) -> None:
    readme = output_root / "README.md"
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text(
        "\n".join(
            [
                "# ONCERCO Backup Metadata Package",
                "",
                "This folder contains CSV exports and summary reports generated from the read-only ONCERCO backup spreadsheets.",
                "",
                "## Contents",
                "",
                "- `csv/`: per-sheet CSV exports",
                "- `manifests/backup_metadata_inventory.csv`: workbook and sheet inventory",
                "- `reports/backup_metadata_summary.md`: concise interpretation of the measurement tables",
                "",
                "## Purpose",
                "",
                "Use this package to inspect disease scores, leaf inclination angle, leaf area index, leaf chlorophyll content, and trial-planning metadata without reopening the original Excel files.",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    log_path = configure_logging()
    started = time.perf_counter()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    CSV_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    logging.info("Starting ONCERCO backup metadata extraction from %s", args.source_root)

    candidates = sorted(path for path in args.source_root.rglob("*.xlsx") if should_include(path))
    logging.info("Found %d candidate workbooks", len(candidates))

    records: list[dict[str, object]] = []
    for workbook in candidates:
        records.extend(export_workbook(workbook, args.source_root, output_root))

    inventory = pd.DataFrame(records).sort_values(["kind", "workbook", "sheet"])
    inventory.to_csv(MANIFEST_ROOT / "backup_metadata_inventory.csv", index=False)
    write_report(output_root, inventory, log_path)
    write_readme(output_root)

    elapsed = time.perf_counter() - started
    logging.info("Wrote %d sheet exports to %s", len(inventory), output_root)
    logging.info("Log file: %s", log_path)
    logging.info("[PHASE] total: %.2fs", elapsed)


if __name__ == "__main__":
    main()
