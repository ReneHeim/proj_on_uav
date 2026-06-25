#!/usr/bin/env python3
"""Run corrected RedEdge-P preprocessing for a 2025 ONCERCO week."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ONCERCO_ROOT = Path("/run/media/davidem/data/ONCERCO")
DEFAULT_OUTPUT_NAME = "2025_rededgep_no_correction_v3"


def setup_logging(log_file: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )


class Timer:
    def __init__(self, name: str):
        self.name = name
        self.elapsed = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        logging.info("[PHASE] %s: start", self.name)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self.t0
        logging.info("[PHASE] %s: %.1fs", self.name, self.elapsed)


def raw_root(oncerco_root: Path, week: str) -> Path:
    return oncerco_root / "data" / "raw" / "2025" / week / "rededgep"


def default_output_root(oncerco_root: Path, output_name: str) -> Path:
    return oncerco_root / "processing" / "local_odm_projects" / output_name


def default_provenance_root(oncerco_root: Path, output_name: str, kind: str) -> Path:
    return oncerco_root / "_provenance" / kind / output_name


def count_captures(set_dir: Path) -> int:
    return sum(1 for _ in set_dir.glob("**/*_1.tif"))


def discover_sets(root: Path, minimum_captures: int) -> list[dict[str, object]]:
    records = []
    for set_dir in sorted(root.glob("*SET")):
        n = count_captures(set_dir)
        records.append(
            {
                "set": set_dir.name,
                "path": str(set_dir),
                "capture_count": n,
                "selected": n >= minimum_captures,
                "reason": "selected" if n >= minimum_captures else "below_minimum_capture_count",
            }
        )
    return records


def run_set(args: argparse.Namespace, week: str, record: dict[str, object]) -> dict[str, object]:
    set_name = str(record["set"])
    set_path = Path(str(record["path"]))
    outdir = args.output_root / week / set_name / "preprocessed_stacks"
    set_log_dir = args.log_root / week / set_name

    cmd = [
        sys.executable,
        "-m",
        "scripts.micasense_rededgep_preprocess",
        "--input-set",
        str(set_path),
        "--panel-set",
        str(set_path),
        "--outdir",
        str(outdir),
        "--panel-strategy",
        args.panel_strategy,
        "--auto-alignment-seed",
        "--alignment-candidate-count",
        str(args.alignment_candidate_count),
        "--alignment-candidates-out",
        str(set_log_dir / "alignment_candidates.json"),
        "--alignment-method",
        args.alignment_method,
        "--alignment-timeout",
        str(args.alignment_timeout),
        "--warp-cache",
        str(set_log_dir / "warps.npz"),
        "--workers",
        str(args.workers),
        "--radiometry-mode",
        args.radiometry_mode,
        "--log-file",
        str(set_log_dir / "preprocess.log"),
        "--profile-out",
        str(set_log_dir / "profile.pstats"),
        "--profile-summary",
        str(set_log_dir / "profile_summary.txt"),
    ]
    if args.refresh_warps:
        cmd.append("--refresh-warps")
    if args.allow_calibrated_fallback:
        cmd.append("--allow-calibrated-fallback")
    if args.include_panchro:
        cmd.append("--include-panchro")
    if args.qa_previews:
        cmd.extend(["--qa-preview-dir", str(set_log_dir / "qa_previews")])
    if args.radiometry_mode == "metashape_compatible":
        if not args.metashape_correction_json:
            raise ValueError("metashape_compatible mode requires --metashape-correction-json")
        cmd.extend(["--metashape-correction-json", str(args.metashape_correction_json)])

    stdout_path = set_log_dir / "preprocess_stdout.log"
    stderr_path = set_log_dir / "preprocess_stderr.log"
    logging.info("[RUN] %s", " ".join(cmd))
    if args.dry_run:
        return {
            **record,
            "returncode": None,
            "elapsed_seconds": 0.0,
            "output_dir": str(outdir),
            "log_dir": str(set_log_dir),
            "command": cmd,
            "dry_run": True,
        }

    set_log_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with stdout_path.open("a") as stdout_handle, stderr_path.open("a") as stderr_handle:
        result = subprocess.run(
            cmd,
            check=False,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
    elapsed = time.perf_counter() - t0
    logging.info("[SET] %s returncode=%s elapsed=%.1fs", set_name, result.returncode, elapsed)
    return {
        **record,
        "returncode": result.returncode,
        "elapsed_seconds": elapsed,
        "output_dir": str(outdir),
        "log_dir": str(set_log_dir),
        "command": cmd,
        "dry_run": False,
    }


def write_outputs(summary: dict, manifest_root: Path) -> tuple[Path, Path]:
    manifest_root.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_root / f"{summary['week']}_rededgep_preprocess_manifest.json"
    report_path = (
        Path("outputs/reports") / f"run_2025_rededgep_preprocess_{summary['week']}_summary.md"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_path.write_text(json.dumps(summary, indent=2))

    rows = [
        "| SET | Raw captures | Selected | Return code | Elapsed s | Output |",
        "|---|---:|---|---:|---:|---|",
    ]
    for record in summary["sets"]:
        rows.append(
            "| {set} | {capture_count} | {selected} | {returncode} | {elapsed_seconds:.1f} | `{output_dir}` |".format(
                **{
                    **record,
                    "returncode": "" if record.get("returncode") is None else record["returncode"],
                    "output_dir": record.get("output_dir", ""),
                    "elapsed_seconds": float(record.get("elapsed_seconds", 0.0)),
                }
            )
        )

    report_path.write_text(
        "\n".join(
            [
                f"## Results: 2025 {summary['week']} RedEdge-P Preprocessing",
                "",
                *rows,
                "",
                (
                    "**Interpretation**: The run uses the corrected core preprocessor, "
                    "which writes `Blue, Green, Red, Red edge, NIR` and stores reflectance "
                    "as uint16 with scale `32767 = 1.0 reflectance`."
                ),
                "",
                "## Outputs",
                "",
                f"- Manifest: `{manifest_path}`",
                f"- Output root: `{summary['output_root']}`",
                f"- Log root: `{summary['log_root']}`",
                "",
                "## Reproducibility",
                "",
                f"- Created UTC: `{summary['created_utc']}`",
                "- Core script: `scripts/preprocessing/rededgep/micasense_rededgep_preprocess.py`",
                "- Output band order: `Blue, Green, Red, Red edge, NIR`",
                f"- Radiometry mode: `{summary['radiometry_mode']}`",
                f"- Alignment method: `{summary['alignment_method']}`",
                f"- Workers: `{summary['workers']}`",
                f"- Panel strategy: `{summary['panel_strategy']}`",
                f"- Dry run: `{summary['dry_run']}`",
            ]
        )
        + "\n"
    )
    return manifest_path, report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week", required=True, help="Week folder name, e.g. week2")
    parser.add_argument("--oncerco-root", type=Path, default=DEFAULT_ONCERCO_ROOT)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--log-root", type=Path)
    parser.add_argument("--manifest-root", type=Path)
    parser.add_argument("--minimum-captures", type=int, default=20)
    parser.add_argument(
        "--panel-strategy",
        choices=("none", "full"),
        default="none",
        help=(
            "Per-capture panel filtering strategy for selected flight SETs. "
            "Default 'none' avoids the slow detect_panels scan; small panel-only "
            "SETs are excluded by --minimum-captures."
        ),
    )
    parser.add_argument("--sets", nargs="+", help="Optional explicit SET names to process.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--alignment-method", choices=("sift", "calibrated"), default="sift")
    parser.add_argument("--alignment-timeout", type=int, default=180)
    parser.add_argument("--alignment-candidate-count", type=int, default=120)
    parser.add_argument("--allow-calibrated-fallback", action="store_true")
    parser.add_argument("--refresh-warps", action="store_true")
    parser.add_argument("--include-panchro", action="store_true")
    parser.add_argument("--qa-previews", action="store_true")
    parser.add_argument(
        "--radiometry-mode",
        choices=("micasense_panel", "micasense_dls", "panel_dls_tie", "metashape_compatible"),
        default="micasense_panel",
    )
    parser.add_argument("--metashape-correction-json", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    output_root = args.output_root or default_output_root(args.oncerco_root, args.output_name)
    log_root = args.log_root or default_provenance_root(args.oncerco_root, args.output_name, "logs")
    manifest_root = args.manifest_root or default_provenance_root(
        args.oncerco_root, args.output_name, "manifests"
    )
    args.output_root = output_root
    args.log_root = log_root
    args.manifest_root = manifest_root

    setup_logging(args.log_file)
    with Timer("discover raw SETs"):
        root = raw_root(args.oncerco_root, args.week)
        if not root.exists():
            raise FileNotFoundError(f"raw RedEdge-P folder not found: {root}")
        records = discover_sets(root, args.minimum_captures)
        if args.sets:
            requested = set(args.sets)
            for record in records:
                record["selected"] = record["set"] in requested
                record["reason"] = "explicitly_selected" if record["selected"] else "not_requested"

    selected = [record for record in records if record["selected"]]
    if not selected:
        raise RuntimeError("no SETs selected for processing")
    logging.info("[SETS] selected=%s", [record["set"] for record in selected])

    run_records = []
    with Timer("process selected SETs"):
        for record in records:
            if record["selected"]:
                run_records.append(run_set(args, args.week, record))
            else:
                run_records.append({**record, "returncode": None, "elapsed_seconds": 0.0})

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "week": args.week,
        "oncerco_root": str(args.oncerco_root),
        "raw_root": str(root),
        "output_root": str(output_root),
        "log_root": str(log_root),
        "manifest_root": str(manifest_root),
        "minimum_captures": args.minimum_captures,
        "panel_strategy": args.panel_strategy,
        "workers": args.workers,
        "alignment_method": args.alignment_method,
        "alignment_timeout": args.alignment_timeout,
        "alignment_candidate_count": args.alignment_candidate_count,
        "allow_calibrated_fallback": args.allow_calibrated_fallback,
        "include_panchro": args.include_panchro,
        "radiometry_mode": args.radiometry_mode,
        "metashape_correction_json": (
            str(args.metashape_correction_json) if args.metashape_correction_json else None
        ),
        "dry_run": args.dry_run,
        "sets": run_records,
    }
    with Timer("write report"):
        manifest_path, report_path = write_outputs(summary, manifest_root)
    logging.info("[DONE] manifest=%s report=%s", manifest_path, report_path)
    return 0 if all(record.get("returncode") in (None, 0) for record in run_records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
