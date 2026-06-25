#!/usr/bin/env bash
set -euo pipefail

cat <<'EOF'
Installing OpenDroneMap command-line support on Arch/CachyOS.

This installs Docker with pacman, enables the Docker service, pulls the official
ODM image, and creates a global /usr/local/bin/odm wrapper.

You will be asked for your sudo password.
EOF

if ! command -v pacman >/dev/null 2>&1; then
  echo "ERROR: pacman was not found. This installer is for Arch/CachyOS." >&2
  exit 2
fi

sudo pacman -Syu --needed docker
sudo systemctl enable --now docker

if ! groups "$USER" | grep -qw docker; then
  sudo usermod -aG docker "$USER"
  added_group=1
else
  added_group=0
fi

sudo tee /usr/local/bin/odm >/dev/null <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  cat <<'USAGE'
Usage:
  odm PROJECT_ROOT PROJECT_NAME [ODM_ARGS...]

PROJECT_ROOT must contain PROJECT_NAME/images/ with the source photos.

Example:
  odm /run/media/davidem/Heim/odm_projects 2025_week0_altum \
    --radiometric-calibration camera --dsm --orthophoto-resolution 0.03

Outputs are written under:
  PROJECT_ROOT/PROJECT_NAME/

Common outputs:
  odm_orthophoto/odm_orthophoto.tif
  odm_dem/dsm.tif
  odm_report/report.pdf
USAGE
  exit 0
fi

project_root=$(realpath "$1")
project_name=$2
shift 2

if [[ ! -d "$project_root/$project_name/images" ]]; then
  echo "ERROR: missing image directory: $project_root/$project_name/images" >&2
  exit 2
fi

docker run --rm -it \
  -u "$(id -u):$(id -g)" \
  -v "$project_root:/datasets" \
  opendronemap/odm:latest \
  "$project_name" \
  --project-path /datasets \
  "$@"
EOF

sudo chmod 755 /usr/local/bin/odm

docker pull opendronemap/odm:latest

cat <<EOF

ODM system install finished.

Test command:
  odm --help

For MicaSense/Altum reflectance processing, include:
  --radiometric-calibration camera

EOF

if [[ $added_group -eq 1 ]]; then
  cat <<'EOF'
Your user was added to the docker group. Log out and back in before running
Docker without sudo. Until then, either restart your session or run:
  newgrp docker
EOF
fi
