#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
FORCE=0

usage() {
  cat <<'EOF'
Usage: bash ckpt.sh [--force]

Downloads the minimum official checkpoints required for ReferDINO Swin-B
reproduction on DAVIS and MeViS.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

mkdir -p "$REPO_ROOT/pretrained" "$REPO_ROOT/ckpt"

download_file() {
  local url="$1"
  local dest="$2"
  local tmp_path="${dest}.part"

  if [[ -f "$dest" && "$FORCE" -eq 0 ]]; then
    echo "Using existing file: $dest"
    return 0
  fi

  rm -f "$tmp_path"
  echo "Downloading $(basename "$dest")"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 --retry-delay 5 -o "$tmp_path" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$tmp_path" "$url"
  else
    echo "Neither curl nor wget is available." >&2
    exit 1
  fi

  mv "$tmp_path" "$dest"
}

download_file \
  "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth" \
  "$REPO_ROOT/pretrained/groundingdino_swinb_cogcoor.pth"

download_file \
  "https://huggingface.co/liangtm/referdino/resolve/main/ryt_swinb.pth" \
  "$REPO_ROOT/ckpt/ryt_swinb.pth"

download_file \
  "https://huggingface.co/liangtm/referdino/resolve/main/mevis_swinb.pth" \
  "$REPO_ROOT/ckpt/mevis_swinb.pth"

echo "Checkpoint setup complete."
