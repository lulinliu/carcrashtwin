#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_10uq.sh --videos-list VLIST --prompts-list PLIST --out OUTPUT_ROOT --model MODEL_SIZE --cond COND_FRAMES
                   [--extra "ARGS"] [--task-idx K] [--resume] [--timeout SEC] [--retries N]

Required:
  --videos-list   当前 task 使用的视频列表 txt（50 行）
  --prompts-list  当前 task 使用的 prompt 列表 txt（50 行）
  --out           输出根目录
  --model         model_size (e.g., 2B)
  --cond          num_conditional_frames (e.g., 5)

Optional:
  --extra         追加参数字符串（可选，原样拼接到命令行末尾）
  --task-idx      仅用于日志标记（可选）
  --resume        已存在输出则跳过
  --timeout SEC   单条样本运行超时秒数（默认 0=不设超时）
  --retries N     失败重试次数（默认 0）
EOF
}

# ---------- args ----------
VLIST=""; PLIST=""; OUTPUT_ROOT=""; MODEL_SIZE=""; COND_FRAMES=""
EXTRA_ARGS=""; TASK_IDX=""
RESUME=false
TIMEOUT=0
RETRIES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --videos-list) VLIST="$2"; shift 2;;
    --prompts-list) PLIST="$2"; shift 2;;
    --out) OUTPUT_ROOT="$2"; shift 2;;
    --model) MODEL_SIZE="$2"; shift 2;;
    --cond) COND_FRAMES="$2"; shift 2;;
    --extra) EXTRA_ARGS="${2:-}"; shift 2;;
    --task-idx) TASK_IDX="$2"; shift 2;;
    --resume) RESUME=true; shift;;
    --timeout) TIMEOUT="${2}"; shift 2;;
    --retries) RETRIES="${2}"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "[ERROR] Unknown arg: $1"; usage; exit 1;;
  esac
done

[[ -f "$VLIST" ]] || { echo "[ERROR] --videos-list not found: $VLIST"; exit 1; }
[[ -f "$PLIST" ]] || { echo "[ERROR] --prompts-list not found: $PLIST"; exit 1; }
[[ -n "$OUTPUT_ROOT" ]] || { echo "[ERROR] --out required"; exit 1; }
[[ -n "$MODEL_SIZE" ]] || { echo "[ERROR] --model required"; exit 1; }
[[ -n "$COND_FRAMES" ]] || { echo "[ERROR] --cond required"; exit 1; }

# ---------- read lists (preserve inner spaces, trim ends) ----------
# NOTE: do not use default IFS splitting; use read -r to keep backslashes & spaces
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

# ---------- setup output & logs ----------
mkdir -p "$OUTPUT_ROOT"
LOG_DIR="${OUTPUT_ROOT%/}/logs_task${TASK_IDX:-NA}"
mkdir -p "$LOG_DIR"
SUMMARY_CSV="${OUTPUT_ROOT%/}/summary_task${TASK_IDX:-NA}.csv"
if [[ ! -f "$SUMMARY_CSV" ]]; then
  echo "idx,input,output,status,tries,secs" > "$SUMMARY_CSV"
fi

echo "[INFO] task=${TASK_IDX:-NA}  num_pairs=$NV  model=$MODEL_SIZE cond=$COND_FRAMES"
echo "[INFO] out=$OUTPUT_ROOT  resume=$RESUME  timeout=$TIMEOUT  retries=$RETRIES"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}  HOST=$HOSTNAME"

# Quick environment probe (non-fatal)
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
python -c "import sys; print('Python', sys.version)" || true
python -c "import torch; print('Torch', torch.__version__, 'CUDA', torch.version.cuda, 'is_available', torch.cuda.is_available())" || true
python -c "import importlib; print('examples.video2world:', importlib.util.find_spec('examples'))" || true

# handle Ctrl-C gracefully
trap 'echo "[ABORT] Caught signal, exiting..."; exit 130' INT TERM

# ---------- runner ----------
run_one() {
  local input="$1"
  local prompt="$2"
  local out_file="$3"
  local item_idx="$4"

  local tries=0
  local start_ts end_ts secs status
  start_ts=$(date +%s)

  if $RESUME && [[ -s "$out_file" ]]; then
    echo "[SKIP][${item_idx}/${NV}] exists: $out_file"
    status="skip"
    end_ts=$(date +%s); secs=$(( end_ts - start_ts ))
    echo "${item_idx},${input},${out_file},${status},${tries},${secs}" >> "$SUMMARY_CSV"
    return 0
  fi

  mkdir -p "$(dirname "$out_file")"
  local logfile="${LOG_DIR}/item_${item_idx}.log"

  # build command
  CMD=( python -m examples.video2world
        --model_size "$MODEL_SIZE"
        --input_path "$input"
        --num_conditional_frames "$COND_FRAMES"
        --prompt "$prompt"
        --save_path "$out_file"
  )
  if [[ -n "$EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARR=( $EXTRA_ARGS )
    CMD+=( "${EXTRA_ARR[@]}" )
  fi

  echo "[RUN][task ${TASK_IDX:-NA}][${item_idx}/${NV}] ${CMD[*]}"
  : > "$logfile"

  while :; do
    ((tries+=1))
    if [[ "$TIMEOUT" -gt 0 ]]; then
      if timeout "$TIMEOUT" "${CMD[@]}" &>> "$logfile"; then
        status="ok"; break
      else
        status="fail-timeout"
      fi
    else
      if "${CMD[@]}" &>> "$logfile"; then
        status="ok"; break
      else
        status="fail"
      fi
    fi
    if (( tries > RETRIES )); then
      break
    fi
    echo "[RETRY] item ${item_idx} attempt ${tries}/${RETRIES}" | tee -a "$logfile"
    sleep 2
  done

  end_ts=$(date +%s); secs=$(( end_ts - start_ts ))
  echo "${item_idx},${input},${out_file},${status},${tries},${secs}" >> "$SUMMARY_CSV"

  if [[ "$status" != "ok" ]]; then
    echo "[ERR][${item_idx}] see log: $logfile"
    return 1
  fi
  return 0
}

# ---------- main loop ----------
fail_ct=0
for ((i=0; i<NV; i++)); do
  INPUT="${VIDEOS[$i]}"
  PROMPT="${PROMPTS[$i]}"

  if [[ -z "$INPUT" || -z "$PROMPT" ]]; then
    echo "[WARN] Skip empty line at $i"
    echo "${i},${INPUT},,empty,0,0" >> "$SUMMARY_CSV"
    continue
  fi
  if [[ ! -f "$INPUT" ]]; then
    echo "[WARN] Video not found, skip: $INPUT"
    echo "${i},${INPUT},,missing,0,0" >> "$SUMMARY_CSV"
    continue
  fi

  fname="$(basename -- "$INPUT")"
  stem="${fname%.*}"
  ext="${fname##*.}"
  out_file="${OUTPUT_ROOT%/}/${stem}_generated.${ext}"

  if ! run_one "$INPUT" "$PROMPT" "$out_file" "$i"; then
    ((fail_ct+=1))
  fi
done

echo "[DONE] task ${TASK_IDX:-NA} finished. Outputs => $OUTPUT_ROOT  fails=$fail_ct"
exit $(( fail_ct > 0 ))
