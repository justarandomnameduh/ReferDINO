#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_DATASET_ROOT="$(cd "$REPO_ROOT/../.." && pwd)/dataset"
DATASET_ROOT="$DEFAULT_DATASET_ROOT"

usage() {
  cat <<'EOF'
Usage: bash tools/prepare_local_rvos_data.sh [--dataset-root /path/to/dataset]

Builds the local ReferDINO data layout for DAVIS and MeViS from the sibling
dataset tree used by the outer segmentation repo.

For inference-only DAVIS reproduction, `davis17/train` is optional. When it is
missing, the script still prepares the `valid` split and metadata required by
`eval/inference_davis.py`.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root)
      [[ $# -ge 2 ]] || { echo "Missing value for --dataset-root" >&2; exit 1; }
      DATASET_ROOT="$2"
      shift 2
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

DATASET_ROOT="$(cd "$DATASET_ROOT" && pwd)"

require_path() {
  local path="$1"
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 1; }
}

replace_with_symlink() {
  local src="$1"
  local dst="$2"
  rm -rf "$dst"
  mkdir -p "$(dirname "$dst")"
  ln -s "$src" "$dst"
}

warn() {
  echo "Warning: $*" >&2
}

extract_split_images() {
  local tar_path="$1"
  local split_dir="$2"

  require_path "$tar_path"
  if [[ -d "$split_dir/JPEGImages" ]] && find "$split_dir/JPEGImages" -mindepth 1 -print -quit >/dev/null; then
    return 0
  fi

  rm -rf "$split_dir/JPEGImages"
  mkdir -p "$split_dir"
  tar -xf "$tar_path" -C "$split_dir"
}

prepare_mevis_split() {
  local split="$1"
  local src_dir="$DATASET_ROOT/mevis/$split"
  local dst_dir="$REPO_ROOT/data/mevis/$split"
  local meta_name

  require_path "$src_dir"
  require_path "$src_dir/mask_dict.json"
  if [[ "$split" == "valid" ]]; then
    meta_name="meta_expressions_v2_release.json"
  else
    meta_name="meta_expressions_v2.json"
  fi
  require_path "$src_dir/$meta_name"

  mkdir -p "$dst_dir"
  extract_split_images "$src_dir/JPEGImages.tar" "$dst_dir"
  replace_with_symlink "$src_dir/mask_dict.json" "$dst_dir/mask_dict.json"
  replace_with_symlink "$src_dir/$meta_name" "$dst_dir/meta_expressions.json"
}

require_path "$DATASET_ROOT/davis17/valid"
require_path "$DATASET_ROOT/davis17/meta_expressions/valid/meta_expressions.json"
require_path "$DATASET_ROOT/davis17_raw/DAVIS"

mkdir -p "$REPO_ROOT/data"
if [[ -e "$DATASET_ROOT/davis17/train" ]]; then
  replace_with_symlink "$DATASET_ROOT/davis17/train" "$REPO_ROOT/data/ref_davis/train"
else
  warn "Skipping DAVIS train split because $DATASET_ROOT/davis17/train is missing. This is fine for inference-only reproduction."
fi
replace_with_symlink "$DATASET_ROOT/davis17/valid" "$REPO_ROOT/data/ref_davis/valid"
replace_with_symlink "$DATASET_ROOT/davis17/meta_expressions" "$REPO_ROOT/data/ref_davis/meta_expressions"
replace_with_symlink "$DATASET_ROOT/davis17_raw/DAVIS" "$REPO_ROOT/data/ref_davis/DAVIS"

prepare_mevis_split "valid"
prepare_mevis_split "valid_u"

echo "ReferDINO local data layout is ready under $REPO_ROOT/data"
