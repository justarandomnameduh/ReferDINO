#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

DATASET_ROOT="${DATASET_ROOT:-/scratch/user/uqqnguy9/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/output}"
VERSION="${VERSION:-swinb_repro}"
MEVIS_SPLIT="${MEVIS_SPLIT:-valid_u}"
NUM_GPUS="${NUM_GPUS:-1}"
OVERLAY_VIDEO_FIRST_N="${OVERLAY_VIDEO_FIRST_N:-5}"
RUN_CKPT=1
RUN_PREPARE=1

usage() {
  cat <<EOF
Usage: bash referdino_inference.sh [options]

Runs full ReferDINO DAVIS and MeViS inference, evaluates both datasets, and
generates a markdown report.

Options:
  --dataset-root PATH         Dataset root to use. Default: $DATASET_ROOT
  --output-root PATH          Output root. Default: $OUTPUT_ROOT
  --version NAME              Output version name. Default: $VERSION
  --mevis-split NAME          MeViS split. Default: $MEVIS_SPLIT
  --num-gpus N                Number of GPUs/processes. Default: $NUM_GPUS
  --overlay-video-first-n N   Number of overlay videos to export. Default: $OVERLAY_VIDEO_FIRST_N
  --skip-ckpt                 Skip checkpoint download step.
  --skip-prepare              Skip dataset layout preparation.
  -h, --help                  Show this help.

Notes:
  - Run this on Bunya with the referdino environment already activated.
  - Every new Bunya session must reload CUDA first:
      module load cuda/11.8.0
EOF
}

note() {
  echo "[referdino] $*"
}

die() {
  echo "[referdino] ERROR: $*" >&2
  exit 1
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || die "Missing required file: $path"
}

require_dir() {
  local path="$1"
  [[ -d "$path" ]] || die "Missing required directory: $path"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --version)
      VERSION="$2"
      shift 2
      ;;
    --mevis-split)
      MEVIS_SPLIT="$2"
      shift 2
      ;;
    --num-gpus)
      NUM_GPUS="$2"
      shift 2
      ;;
    --overlay-video-first-n)
      OVERLAY_VIDEO_FIRST_N="$2"
      shift 2
      ;;
    --skip-ckpt)
      RUN_CKPT=0
      shift
      ;;
    --skip-prepare)
      RUN_PREPARE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

LOG_DIR="$OUTPUT_ROOT/logs/$VERSION"
REPORT_DIR="$OUTPUT_ROOT/reports"
mkdir -p "$LOG_DIR" "$REPORT_DIR"

note "Repo root: $REPO_ROOT"
note "Dataset root: $DATASET_ROOT"
note "Output root: $OUTPUT_ROOT"
note "Version: $VERSION"
note "MeViS split: $MEVIS_SPLIT"

require_dir "$DATASET_ROOT"

if ! command -v python >/dev/null 2>&1; then
  die "python is not available in PATH"
fi

if ! command -v nvcc >/dev/null 2>&1; then
  die "nvcc is not available. On Bunya, run 'module load cuda/11.8.0' in every new session."
fi

python - <<'PY'
import sys

try:
    import torch
except Exception as exc:
    raise SystemExit(f"PyTorch import failed: {exc}")

print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is not active for this shell. On Bunya, run 'module load cuda/11.8.0' "
        "and reactivate the referdino environment before rerunning."
    )
PY

REQUIRED_CHECKPOINTS=(
  "$REPO_ROOT/pretrained/groundingdino_swinb_cogcoor.pth"
  "$REPO_ROOT/ckpt/ryt_swinb.pth"
  "$REPO_ROOT/ckpt/mevis_swinb.pth"
)

missing_ckpts=0
for ckpt in "${REQUIRED_CHECKPOINTS[@]}"; do
  if [[ ! -f "$ckpt" ]]; then
    missing_ckpts=1
    break
  fi
done

if [[ "$missing_ckpts" -eq 1 ]]; then
  if [[ "$RUN_CKPT" -eq 1 ]]; then
    note "Downloading missing checkpoints"
    bash "$REPO_ROOT/ckpt.sh" | tee "$LOG_DIR/ckpt.log"
  else
    die "Checkpoint files are missing and --skip-ckpt was provided"
  fi
fi

for ckpt in "${REQUIRED_CHECKPOINTS[@]}"; do
  require_file "$ckpt"
done

if [[ "$RUN_PREPARE" -eq 1 ]]; then
  note "Preparing local DAVIS and MeViS data layout"
  bash "$REPO_ROOT/tools/prepare_local_rvos_data.sh" --dataset-root "$DATASET_ROOT" \
    | tee "$LOG_DIR/prepare_data.log"
fi

require_file "$REPO_ROOT/data/ref_davis/meta_expressions/valid/meta_expressions.json"
require_dir "$REPO_ROOT/data/ref_davis/valid/JPEGImages"
require_file "$REPO_ROOT/data/mevis/$MEVIS_SPLIT/meta_expressions.json"
require_file "$REPO_ROOT/data/mevis/$MEVIS_SPLIT/mask_dict.json"
require_dir "$REPO_ROOT/data/mevis/$MEVIS_SPLIT/JPEGImages"

note "Running full DAVIS inference"
python "$REPO_ROOT/eval/inference_davis.py" \
  -c "$REPO_ROOT/configs/davis_swinb.yaml" \
  -ng "$NUM_GPUS" \
  -ckpt "$REPO_ROOT/ckpt/ryt_swinb.pth" \
  --version "$VERSION" \
  --overlay_video_first_n "$OVERLAY_VIDEO_FIRST_N" \
  | tee "$LOG_DIR/davis_inference.log"

note "Evaluating DAVIS predictions"
python "$REPO_ROOT/eval/eval_davis.py" \
  --results_path "$OUTPUT_ROOT/davis/$VERSION/Annotations" \
  | tee "$LOG_DIR/davis_eval.log"

note "Running full MeViS inference"
python "$REPO_ROOT/eval/inference_mevis.py" \
  --split "$MEVIS_SPLIT" \
  -c "$REPO_ROOT/configs/mevis_swinb.yaml" \
  -ng "$NUM_GPUS" \
  -ckpt "$REPO_ROOT/ckpt/mevis_swinb.pth" \
  --version "$VERSION" \
  --overlay_video_first_n "$OVERLAY_VIDEO_FIRST_N" \
  | tee "$LOG_DIR/mevis_inference.log"

MEVIS_EVAL_JSON="eval_results_${MEVIS_SPLIT}.json"

note "Evaluating MeViS predictions"
python "$REPO_ROOT/eval/eval_mevis.py" \
  --mevis_pred_path "$OUTPUT_ROOT/mevis/$VERSION/$MEVIS_SPLIT/Annotations" \
  --mevis_exp_path "$REPO_ROOT/data/mevis/$MEVIS_SPLIT/meta_expressions.json" \
  --mevis_mask_path "$REPO_ROOT/data/mevis/$MEVIS_SPLIT/mask_dict.json" \
  --save_name "$MEVIS_EVAL_JSON" \
  | tee "$LOG_DIR/mevis_eval.log"

REPORT_PATH="$REPORT_DIR/${VERSION}_report.md"
SUMMARY_JSON_PATH="$REPORT_DIR/${VERSION}_summary.json"

note "Generating report at $REPORT_PATH"
python - "$OUTPUT_ROOT" "$VERSION" "$MEVIS_SPLIT" "$LOG_DIR" "$REPORT_PATH" "$SUMMARY_JSON_PATH" <<'PY'
import csv
import datetime as dt
import json
import os
import statistics
import sys

output_root, version, mevis_split, log_dir, report_path, summary_json_path = sys.argv[1:]

targets = {
    "davis": {"J": 65.1, "F": 72.9, "J&F": 68.9},
    "mevis": {"J": 44.7, "F": 53.9, "J&F": 49.3},
}

davis_rows = []
for anno_idx in range(4):
    csv_path = os.path.join(
        output_root,
        "davis",
        version,
        "Annotations",
        f"anno_{anno_idx}",
        "global_results-val.csv",
    )
    if not os.path.exists(csv_path):
        raise SystemExit(f"Missing DAVIS evaluation CSV: {csv_path}")
    with open(csv_path, newline="") as handle:
        row = next(csv.DictReader(handle))
    davis_rows.append(
        {
            "annotator": anno_idx,
            "J&F": float(row["J&F-Mean"]) * 100.0,
            "J": float(row["J-Mean"]) * 100.0,
            "F": float(row["F-Mean"]) * 100.0,
        }
    )

davis_summary = {
    metric: statistics.mean(row[metric] for row in davis_rows)
    for metric in ("J", "F", "J&F")
}

mevis_json_path = os.path.join(output_root, "mevis", version, mevis_split, f"eval_results_{mevis_split}.json")
if not os.path.exists(mevis_json_path):
    raise SystemExit(f"Missing MeViS evaluation JSON: {mevis_json_path}")

with open(mevis_json_path) as handle:
    mevis_data = json.load(handle)

mevis_j = [float(values[0]) * 100.0 for values in mevis_data.values()]
mevis_f = [float(values[1]) * 100.0 for values in mevis_data.values()]
if not mevis_j or not mevis_f:
    raise SystemExit("MeViS evaluation JSON is empty")

mevis_summary = {
    "J": statistics.mean(mevis_j),
    "F": statistics.mean(mevis_f),
    "J&F": (statistics.mean(mevis_j) + statistics.mean(mevis_f)) / 2.0,
}

summary = {
    "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
    "version": version,
    "output_root": output_root,
    "log_dir": log_dir,
    "davis": {
        "metrics": davis_summary,
        "targets": targets["davis"],
        "per_annotator": davis_rows,
    },
    "mevis": {
        "split": mevis_split,
        "metrics": mevis_summary,
        "targets": targets["mevis"],
        "num_expressions": len(mevis_data),
        "eval_json": mevis_json_path,
    },
}

for dataset_name in ("davis", "mevis"):
    metrics = summary[dataset_name]["metrics"]
    dataset_targets = summary[dataset_name]["targets"]
    summary[dataset_name]["delta"] = {
        key: metrics[key] - dataset_targets[key] for key in ("J", "F", "J&F")
    }

with open(summary_json_path, "w") as handle:
    json.dump(summary, handle, indent=2)

def fmt(value):
    return f"{value:.2f}"

lines = [
    "# ReferDINO Reproduction Report",
    "",
    f"- Generated: {summary['generated_at']}",
    f"- Version: `{version}`",
    f"- Output root: `{output_root}`",
    f"- Logs: `{log_dir}`",
    "",
    "## DAVIS",
    "",
    "| Metric | Observed | Paper Target | Delta |",
    "| --- | ---: | ---: | ---: |",
]

for key in ("J", "F", "J&F"):
    lines.append(
        f"| {key} | {fmt(summary['davis']['metrics'][key])} | "
        f"{fmt(summary['davis']['targets'][key])} | {fmt(summary['davis']['delta'][key])} |"
    )

lines.extend(
    [
        "",
        "Per-annotator DAVIS summary:",
        "",
        "| Annotator | J | F | J&F |",
        "| --- | ---: | ---: | ---: |",
    ]
)

for row in davis_rows:
    lines.append(
        f"| anno_{row['annotator']} | {fmt(row['J'])} | {fmt(row['F'])} | {fmt(row['J&F'])} |"
    )

lines.extend(
    [
        "",
        "## MeViS",
        "",
        f"- Split: `{mevis_split}`",
        f"- Expressions evaluated: {summary['mevis']['num_expressions']}",
        f"- Raw evaluation JSON: `{mevis_json_path}`",
        "",
        "| Metric | Observed | Paper Target | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
)

for key in ("J", "F", "J&F"):
    lines.append(
        f"| {key} | {fmt(summary['mevis']['metrics'][key])} | "
        f"{fmt(summary['mevis']['targets'][key])} | {fmt(summary['mevis']['delta'][key])} |"
    )

lines.extend(
    [
        "",
        "## Artifacts",
        "",
        f"- DAVIS predictions: `{os.path.join(output_root, 'davis', version, 'Annotations')}`",
        f"- DAVIS overlays: `{os.path.join(output_root, 'davis', version, 'overlay_videos')}`",
        f"- MeViS predictions: `{os.path.join(output_root, 'mevis', version, mevis_split, 'Annotations')}`",
        f"- MeViS overlays: `{os.path.join(output_root, 'mevis', version, mevis_split, 'overlay_videos')}`",
        f"- JSON summary: `{summary_json_path}`",
    ]
)

with open(report_path, "w") as handle:
    handle.write("\n".join(lines) + "\n")

print(f"Report written to {report_path}")
print(f"JSON summary written to {summary_json_path}")
PY

note "Full reproduction complete"
note "Report: $REPORT_PATH"
note "JSON summary: $SUMMARY_JSON_PATH"
