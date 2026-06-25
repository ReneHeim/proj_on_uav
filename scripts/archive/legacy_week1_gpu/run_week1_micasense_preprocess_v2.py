#!/usr/bin/env python3
"""Run corrected RedEdge-P preprocessing for all 2025 week1 SETs."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

RAW_ROOT = Path("/mnt/data/ONCERCO/data/raw/2025/week1/rededgep")
ORTHORITY_ROOT = Path("/mnt/data/ONCERCO/processing/local_odm_projects/week1_orthority_per_image")
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/data/ONCERCO/processing/local_odm_projects/2025_rededgep_no_correction_v2"
)
DEFAULT_LOG_ROOT = Path(
    "/mnt/data/ONCERCO/_provenance/logs/processing_2025_rededgep_no_correction_v2"
)
DEFAULT_MANIFEST_ROOT = Path(
    "/mnt/data/ONCERCO/_provenance/manifests/processing_2025_rededgep_no_correction_v2"
)
SETS = ("0000SET", "0001SET", "0002SET", "0003SET", "0004SET")
ORTHORITY_OUTPUT_DIRS = (
    "orthophotos",
    "orthophotos_retry_uncropped_dsm",
    "sparse_dsm_recovery/orthophotos_retry_sparse_dsm",
)


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


def timed(name: str):
    class Timer:
        def __enter__(self):
            self.t0 = time.perf_counter()
            logging.info("[PHASE] %s: start", name)
            return self

        def __exit__(self, exc_type, exc, tb):
            self.elapsed = time.perf_counter() - self.t0
            logging.info("[PHASE] %s: %.1fs", name, self.elapsed)

    return Timer()


def available_capture_names(set_name: str) -> list[str]:
    set_dir = ORTHORITY_ROOT / set_name
    names: set[str] = set()
    for rel_dir in ORTHORITY_OUTPUT_DIRS:
        folder = set_dir / rel_dir
        if not folder.exists():
            continue
        for tif_path in folder.glob("IMG_*_6.tif"):
            names.add(tif_path.name.replace("_6.tif", "_1.tif"))
    return sorted(names)


def raw_capture_count(set_name: str) -> int:
    return sum(1 for _ in (RAW_ROOT / set_name).glob("**/*_1.tif"))


def write_capture_list(set_name: str, capture_names: list[str], manifest_root: Path) -> Path:
    out = manifest_root / f"week1_{set_name}_capture_list.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(capture_names) + "\n")
    return out


def run_preprocess_for_set(args: argparse.Namespace, set_name: str, capture_list: Path) -> dict:
    set_log_dir = args.log_root / "week1" / set_name
    outdir = args.output_root / "week1" / set_name / "preprocessed_stacks"
    set_log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/micasense_rededgep_preprocess.py",
        "--input-set",
        str(RAW_ROOT / set_name),
        "--panel-set",
        str(RAW_ROOT / set_name),
        "--capture-list",
        str(capture_list),
        "--outdir",
        str(outdir),
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
        "--log-file",
        str(set_log_dir / "preprocess.log"),
        "--profile-out",
        str(set_log_dir / "profile.pstats"),
        "--profile-summary",
        str(set_log_dir / "profile_summary.txt"),
    ]
    if args.allow_calibrated_fallback:
        cmd.append("--allow-calibrated-fallback")
    if args.include_panchro:
        cmd.append("--include-panchro")
    if args.qa_previews:
        cmd.extend(["--qa-preview-dir", str(set_log_dir / "qa_previews")])

    child_stdout = set_log_dir / "preprocess_stdout.log"
    child_stderr = set_log_dir / "preprocess_stderr.log"
    logging.info("[RUN] %s", " ".join(cmd))
    logging.info("[RUN] stdout=%s stderr=%s", child_stdout, child_stderr)
    t0 = time.perf_counter()
    with child_stdout.open("a") as stdout_handle, child_stderr.open("a") as stderr_handle:
        result = subprocess.run(
            cmd,
            check=False,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
    elapsed = time.perf_counter() - t0
    logging.info("[SET] %s returncode=%s elapsed=%.1fs", set_name, result.returncode, elapsed)
    return {
        "set": set_name,
        "returncode": result.returncode,
        "elapsed_seconds": elapsed,
        "output_dir": str(outdir),
        "log_dir": str(set_log_dir),
        "command": cmd,
    }


def write_report(summary: dict, manifest_root: Path) -> None:
    report = manifest_root / "week1_preprocess_v2_summary.md"
    rows = [
        "| SET | Raw captures | Clean capture list | Return code | Elapsed s | Output |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for record in summary["sets"]:
        rows.append(
            "| {set} | {raw_capture_count} | {capture_list_count} | {returncode} | "
            "{elapsed_seconds:.1f} | `{output_dir}` |".format(**record)
        )
    outputs = "\n".join(f"- `{record['output_dir']}`" for record in summary["sets"])
    report.write_text(
        "\n".join(
            [
                "## Results: Week1 RedEdge-P Preprocessing V2",
                "",
                *rows,
                "",
                (
                    "**Interpretation**: The corrected run uses only captures with available "
                    "Orthority products, excluding panel/failed frames from the stack inputs."
                ),
                "",
                f"**Outputs**:\n{outputs}",
                "",
                "## Reproducibility",
                "",
                f"- Created UTC: `{summary['created_utc']}`",
                f"- Script: `scripts/micasense_rededgep_preprocess.py`",
                f"- Runner: `scripts/run_week1_micasense_preprocess_v2.py`",
                f"- Workers: `{summary['workers']}`",
                f"- Alignment method: `{summary['alignment_method']}`",
                f"- Alignment timeout: `{summary['alignment_timeout']}`",
                f"- Calibrated fallback: `{summary['allow_calibrated_fallback']}`",
                f"- Capture-list source: `{ORTHORITY_ROOT}`",
            ]
        )
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--alignment-method", choices=("sift", "calibrated"), default="sift")
    parser.add_argument("--alignment-timeout", type=int, default=180)
    parser.add_argument("--alignment-candidate-count", type=int, default=120)
    parser.add_argument("--allow-calibrated-fallback", action="store_true")
    parser.add_argument("--include-panchro", action="store_true")
    parser.add_argument("--qa-previews", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()

    setup_logging(args.log_file)
    args.manifest_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workers": args.workers,
        "alignment_method": args.alignment_method,
        "alignment_timeout": args.alignment_timeout,
        "allow_calibrated_fallback": args.allow_calibrated_fallback,
        "output_root": str(args.output_root),
        "log_root": str(args.log_root),
        "manifest_root": str(args.manifest_root),
        "sets": [],
    }

    with timed("build capture lists"):
        for set_name in SETS:
            capture_names = available_capture_names(set_name)
            capture_list = write_capture_list(set_name, capture_names, args.manifest_root)
            record = {
                "set": set_name,
                "raw_capture_count": raw_capture_count(set_name),
                "capture_list_count": len(capture_names),
                "capture_list": str(capture_list),
            }
            logging.info(
                "[CAPTURE-LIST] %s raw=%s selected=%s list=%s",
                set_name,
                record["raw_capture_count"],
                record["capture_list_count"],
                capture_list,
            )
            summary["sets"].append(record)

    if not args.dry_run:
        for record in summary["sets"]:
            with timed(f"preprocess {record['set']}"):
                result = run_preprocess_for_set(args, record["set"], Path(record["capture_list"]))
            record.update(result)
    else:
        for record in summary["sets"]:
            record.update(
                {
                    "returncode": 0,
                    "elapsed_seconds": 0.0,
                    "output_dir": str(
                        args.output_root / "week1" / record["set"] / "preprocessed_stacks"
                    ),
                    "log_dir": str(args.log_root / "week1" / record["set"]),
                    "dry_run": True,
                }
            )

    with timed("write summary"):
        (args.manifest_root / "week1_preprocess_v2_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        write_report(summary, args.manifest_root)

    failed = [record for record in summary["sets"] if record["returncode"] != 0]
    if failed:
        logging.error("[DONE] failed_sets=%s", [record["set"] for record in failed])
        return 2
    logging.info("[DONE] all SETs completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
