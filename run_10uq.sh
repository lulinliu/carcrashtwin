#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: bash.sh --videos-list VLIST --prompts-list PLIST --out OUTPUT_ROOT --model MODEL_SIZE --cond COND_FRAMES [--extra "ARGS"] [--task-idx K]
  --videos-list   当前 task 使用的视频列表 txt（50 行）
  --prompts-list  当前 task 使用的 prompt 列表 txt（50 行）
  --out           输出根目录
  --model         model_size (e.g., 2B)
  --cond          num_conditional_frames (e.g., 5)
  --extra         追加参数字符串（可选，原样拼接到命令行末尾）
  --task-idx      仅用于日志标记（可选）
EOF
}

VLIST=""
PLIST=""
OUTPUT_ROOT=""
MODEL_SIZE=""
COND_FRAMES=""
EXTRA_ARGS=""
TASK_IDX=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --videos-list) VLIST="$2"; shift 2;;
    --prompts-list) PLIST="$2"; shift 2;;
    --out) OUTPUT_ROOT="$2"; shift 2;;
    --model) MODEL_SIZE="$2"; shift 2;;
    --cond) COND_FRAMES="$2"; shift 2;;
    --extra) EXTRA_ARGS="${2:-}"; shift 2;;
    --task-idx) TASK_IDX="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "[ERROR] Unknown arg: $1"; usage; exit 1;;
  esac
done

[[ -f "$VLIST" ]] || { echo "[ERROR] --videos-list not found: $VLIST"; exit 1; }
[[ -f "$PLIST" ]] || { echo "[ERROR] --prompts-list not found: $PLIST"; exit 1; }
[[ -n "$OUTPUT_ROOT" ]] || { echo "[ERROR] --out required"; exit 1; }
[[ -n "$MODEL_SIZE" ]] || { echo "[ERROR] --model required"; exit 1; }
[[ -n "$COND_FRAMES" ]] || { echo "[ERROR] --cond required"; exit 1; }

mapfile -t VIDEOS < <(sed -e 's/^[[:space:]]*//; s/[[:space:]]*$//' "$VLIST" | grep -v '^[[:space:]]*$')
mapfile -t PROMPTS < <(sed -e 's/^[[:space:]]*//; s/[[:space:]]*$//' "$PLIST" | grep -v '^[[:space:]]*$')

NV=${#VIDEOS[@]}
NP=${#PROMPTS[@]}
if [[ "$NV" -ne "$NP" ]]; then
  echo "[ERROR] Line count mismatch: videos=$NV, prompts=$NP" >&2; exit 1
fi
if [[ "$NV" -eq 0 ]]; then
  echo "[ERROR] Empty lists." >&2; exit 1
fi
if [[ "$NV" -ne 50 ]]; then
  echo "[WARN] Expect 50 lines, got $NV. Continue anyway."
fi

mkdir -p "$OUTPUT_ROOT"
echo "[INFO] task=${TASK_IDX:-NA}  num_pairs=$NV  model=$MODEL_SIZE cond=$COND_FRAMES"

for ((i=0; i<NV; i++)); do
  INPUT="${VIDEOS[$i]}"
  PROMPT="${PROMPTS[$i]}"

  if [[ -z "$INPUT" || -z "$PROMPT" ]]; then
    echo "[WARN] Skip empty line at $i"
    continue
  fi
  if [[ ! -f "$INPUT" ]]; then
    echo "[WARN] Video not found, skip: $INPUT"
    continue
  fi

  fname="$(basename -- "$INPUT")"
  stem="${fname%.*}"
  ext="${fname##*.}"
  out_file="${OUTPUT_ROOT}/${stem}_generated.${ext}"

  CMD=( python -m examples.video2world
        --model_size "$MODEL_SIZE"
        --input_path "$INPUT"
        --num_conditional_frames "$COND_FRAMES"
        --prompt "$PROMPT"
        --save_path "$out_file"
  )

  if [[ -n "$EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARR=( $EXTRA_ARGS )
    CMD+=( "${EXTRA_ARR[@]}" )
  fi

  echo "[RUN][task ${TASK_IDX:-NA}][${i}/${NV}] ${CMD[*]}"
  "${CMD[@]}"
done

echo "[DONE] task ${TASK_IDX:-NA} finished. Outputs => $OUTPUT_ROOT"
