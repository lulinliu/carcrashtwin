#!/bin/bash
#SBATCH --job-name=video2world_10array
#SBATCH -A CGAI24022
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -t 10:00:00
#SBATCH --array=0-9
#SBATCH -o /scratch/10102/hh29499/carcrashtwin/%x_%A_%a.out
#SBATCH -e /scratch/10102/hh29499/carcrashtwin/%x_%A_%a.err

set -euo pipefail

VIDEOS_MANIFEST="${VIDEOS_MANIFEST:-tasks_videos_manifest.txt}"     # 10 lines
PROMPTS_MANIFEST="${PROMPTS_MANIFEST:-tasks_prompts_manifest.txt}"  # 10 lines
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
MODEL_SIZE="${MODEL_SIZE:-2B}"
COND_FRAMES="${COND_FRAMES:-5}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mapfile -t VIDEO_LISTS < <(sed -e 's/^[[:space:]]*//; s/[[:space:]]*$//' "$VIDEOS_MANIFEST" | grep -v '^[[:space:]]*$')
mapfile -t PROMPT_LISTS < <(sed -e 's/^[[:space:]]*//; s/[[:space:]]*$//' "$PROMPTS_MANIFEST" | grep -v '^[[:space:]]*$')

[[ "${#VIDEO_LISTS[@]}" -eq 10 ]] || { echo "Need 10 lines in $VIDEOS_MANIFEST"; exit 1; }
[[ "${#PROMPT_LISTS[@]}" -eq 10 ]] || { echo "Need 10 lines in $PROMPTS_MANIFEST"; exit 1; }

IDX="${SLURM_ARRAY_TASK_ID}"
VLIST="${VIDEO_LISTS[$IDX]}"
PLIST="${PROMPT_LISTS[$IDX]}"

echo "[INFO] node=$HOSTNAME idx=$IDX gpu=$CUDA_VISIBLE_DEVICES"
mkdir -p "$OUTPUT_ROOT"

bash ./run_10uq.sh \
  --videos-list "$VLIST" \
  --prompts-list "$PLIST" \
  --out "$OUTPUT_ROOT" \
  --model "$MODEL_SIZE" \
  --cond "$COND_FRAMES" \
  --extra "$EXTRA_ARGS" \
  --task-idx "$IDX"
