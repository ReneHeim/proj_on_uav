#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat >&2 <<'USAGE'
Usage:
  scripts/run_metashape_rededgep_workflow.sh <week> <out-root> [extra metashape script args...]

Example reproduction run:
  scripts/run_metashape_rededgep_workflow.sh week3 /tmp/metashape_repro_week3 --limit-captures 80 --downscale 2

Example production run:
  scripts/run_metashape_rededgep_workflow.sh week2 /mnt/data/ONCERCO/data/processed/2025/week2/metashape_custom --downscale 1
USAGE
  exit 2
fi

week="$1"
out_root="$2"
shift 2

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
raw_week="/mnt/data/ONCERCO/data/raw/2025/${week}/rededgep"
project="${out_root}/${week}.psx"

if [[ ! -d "${raw_week}" ]]; then
  echo "Raw week folder does not exist: ${raw_week}" >&2
  exit 1
fi

metashape_bin="${METASHAPE_BIN:-}"
if [[ -z "${metashape_bin}" ]]; then
  for candidate in metashape agisoft-metashape Metashape metashape.sh; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      metashape_bin="$(command -v "${candidate}")"
      break
    fi
  done
fi

if [[ -z "${metashape_bin}" ]]; then
  cat >&2 <<EOF
Metashape executable was not found.

Install Agisoft Metashape Professional or set:
  export METASHAPE_BIN=/path/to/metashape.sh

Then rerun:
  $0 ${week} ${out_root} $*
EOF
  exit 127
fi

mkdir -p "${out_root}"
exec "${metashape_bin}" -r "${repo_root}/scripts/preprocessing/rededgep/metashape_process_rededgep_week.py" -- \
  --raw-week "${raw_week}" \
  --out-root "${out_root}" \
  --project "${project}" \
  --mode panels_sun \
  "$@"
