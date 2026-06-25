#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/odm_process_week.sh IMAGE_DIR PROJECT_ROOT PROJECT_NAME [ODM_ARGS...]

Example:
  scripts/odm_process_week.sh \
    /run/media/davidem/Heim/raw/2025/week0/images \
    /run/media/davidem/Heim/odm_projects \
    2025_week0_altum \
    --orthophoto-resolution 0.03

The script runs the official OpenDroneMap Docker image against a project at:
  PROJECT_ROOT/PROJECT_NAME

IMAGE_DIR is mounted read-only as the ODM project's images/ directory, so raw
imagery is not copied. ODM outputs are written under PROJECT_ROOT/PROJECT_NAME.

Set ODM_IMAGE to override the container image. Default: opendronemap/odm:latest
Set ODM_USER_ARGS="" if the image needs to run as root on your system.
EOF
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" || $# -lt 3 ]]; then
  usage
  exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
  cat >&2 <<'EOF'
ERROR: docker is not installed or not on PATH.

Install and start Docker first. On CachyOS/Arch this is typically:
  sudo pacman -S docker docker-compose
  sudo systemctl enable --now docker
  sudo usermod -aG docker "$USER"

Then log out/in, or run the script with sudo until group membership refreshes.
EOF
  exit 127
fi

image_dir=$(realpath "$1")
project_root=$(realpath -m "$2")
project_name=$3
shift 3

if [[ ! -d "$image_dir" ]]; then
  echo "ERROR: IMAGE_DIR does not exist or is not a directory: $image_dir" >&2
  exit 2
fi

first_image=$(find "$image_dir" -type f \( -iname '*.tif' -o -iname '*.tiff' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.dng' \) -print -quit)
if [[ -z "$first_image" ]]; then
  echo "ERROR: IMAGE_DIR contains no TIFF/JPEG/DNG images: $image_dir" >&2
  exit 2
fi

project_dir="$project_root/$project_name"
mkdir -p "$project_dir"

odm_image=${ODM_IMAGE:-opendronemap/odm:latest}
read -r -a odm_user_args <<< "${ODM_USER_ARGS:---user $(id -u):$(id -g)}"

default_args=(
  --project-path /datasets
  --radiometric-calibration camera
  --auto-boundary
)

echo "[ODM] image:        $odm_image"
echo "[ODM] image dir:    $image_dir"
echo "[ODM] project dir:  $project_dir"
echo "[ODM] project name: $project_name"
echo "[ODM] extra args:   $*"

docker run --rm \
  "${odm_user_args[@]}" \
  -v "$project_root:/datasets" \
  -v "$image_dir:/datasets/$project_name/images:ro" \
  "$odm_image" \
  "$project_name" \
  "${default_args[@]}" \
  "$@"

cat <<EOF

[ODM] Done. Key outputs to inspect:
  $project_dir/odm_orthophoto/odm_orthophoto.tif
  $project_dir/odm_dem/dsm.tif
  $project_dir/odm_report/report.pdf
EOF
