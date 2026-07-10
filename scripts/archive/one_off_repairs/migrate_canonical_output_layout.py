#!/usr/bin/env python3
"""Move ignored output directories into the canonical layout.

Run once after a verified filesystem snapshot. New code must use canonical
paths; the migration intentionally leaves no compatibility symlinks.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUTPUTS = ROOT / "outputs"
MANIFEST = OUTPUTS / "provenance" / "output_layout_migration_20260710.csv"


@dataclass(frozen=True)
class Move:
    source: str
    destination: str


def severity_destination(name: str, prefix: str, family: str) -> Move:
    return Move(name, f"runs/analysis/severity/{family}/{name.removeprefix(prefix)}")


MOVES = [
    Move("ARCHIVED", "archive/historical_runs"),
    Move("backup_metadata", "runs/metadata/backup_metadata"),
    Move("code_session_exports", "archive/historical_runs/code_session_exports"),
    Move("cpu_sift_runs", "runs/diagnostics/alignment/cpu_sift_runs"),
    Move("cross_year_generalization_2024_to_2025", "runs/analysis/severity/cross_year/generalization_2024_to_2025"),
    *[
        severity_destination(name, "current_severity_", "current")
        for name in [
            "current_severity_2024_to_2025",
            "current_severity_angular_disorder_2024_to_2025",
            "current_severity_curve_embeddings_2024_to_2025",
            "current_severity_curve_only_elasticnet_gam_2024_to_2025",
            "current_severity_curve_only_functional_2024_to_2025",
            "current_severity_curve_only_hurdle_elasticnet_gam_2024_to_2025",
            "current_severity_curve_only_offnadir_2024_to_2025",
            "current_severity_curve_only_pls_2024_to_2025",
            "current_severity_curve_only_smooth_basis_2024_to_2025",
            "current_severity_design_aware_mixture_2024_to_2025",
            "current_severity_elastic_srvf_shape_2024_to_2025",
            "current_severity_healthy_angular_signature_2024_to_2025",
            "current_severity_healthy_counterfactual_2024_to_2025",
            "current_severity_magnitude_shape_functional_2024_to_2025",
            "current_severity_multiangular_plus_nadir_2024_to_2025",
            "current_severity_raa_geometry_fusion_2024_to_2025",
            "current_severity_sparse_functional_discriminant_shape_2024_to_2025",
            "current_severity_subplots_2024_to_2025",
            "current_severity_subplots_curve_embeddings_2024_to_2025",
        ]
    ],
    Move("diagnostics", "runs/diagnostics"),
    Move("disease", "shared/disease"),
    Move("early_warning_severity_2024", "runs/analysis/early_warning/severity_2024"),
    Move("figures", "archive/legacy_unscoped/figures"),
    *[
        severity_destination(name, "future_severity_", "future")
        for name in [
            "future_severity_current_hurdle_vza_raa_2024_to_2025",
            "future_severity_raa_geometry_residual_2024_to_2025",
            "future_severity_vza_raa_feature_selection_improvement",
        ]
    ],
    Move("logs", "archive/legacy_unscoped/logs"),
    Move("main_compact_multiangular_offnadir_2024_to_2025", "runs/analysis/severity/current/main_compact_multiangular_offnadir_2024_to_2025"),
    Move("manifests", "archive/legacy_unscoped/manifests"),
    Move("manuscript_tables", "deliverables/manuscript"),
    Move("multiangular_distribution_feature_family", "runs/analysis/severity/future/compact_distribution_feature_family"),
    Move("presentation_assets", "deliverables/presentation/assets"),
    Move("quality", "runs/diagnostics/quality"),
    Move("reports", "archive/legacy_unscoped/reports"),
    Move("result_01_raa_sun_geometry", "runs/analysis/reflectance/raa_sun_geometry"),
    Move("result_01_reflectance_distributions", "runs/analysis/reflectance/distributions"),
    Move("result_02_canopy_gap_vza", "runs/analysis/canopy_structure/gap_vza"),
    Move("result_03_vza_curve_shape_metrics", "runs/analysis/sun_geometry/vza_curve_shape_metrics"),
    Move("results", "archive/legacy_unscoped/results"),
    Move("severity_actual_vs_predicted_2024_oof_best_vs_nadir", "deliverables/presentation/severity_actual_vs_predicted_2024_oof"),
    Move("severity_actual_vs_predicted_best_vs_nadir", "deliverables/presentation/severity_actual_vs_predicted_2025"),
    Move("test_cpu_warps", "archive/historical_runs/test_cpu_warps"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="perform moves")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, str]] = []
    for move in MOVES:
        source = OUTPUTS / move.source
        destination = OUTPUTS / move.destination
        if source.is_symlink():
            status = "already_migrated"
        elif not source.exists():
            status = "missing"
        elif destination.exists():
            conflicts = [child.name for child in source.iterdir() if (destination / child.name).exists()]
            if conflicts:
                status = f"destination_conflict:{','.join(conflicts)}"
            elif args.apply and source.is_dir() and destination.is_dir():
                for child in source.iterdir():
                    child.rename(destination / child.name)
                source.rmdir()
                status = "merged"
            else:
                status = "planned_merge"
        elif args.apply:
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
            status = "moved"
        else:
            status = "planned"
        rows.append({"old_path": f"outputs/{move.source}", "new_path": f"outputs/{move.destination}", "status": status})

    if args.apply:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["old_path", "new_path", "status"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {MANIFEST.relative_to(ROOT)}")
    for row in rows:
        print(f"{row['status']:36} {row['old_path']} -> {row['new_path']}")


if __name__ == "__main__":
    main()
