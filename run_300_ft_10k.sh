#!/usr/bin/env bash
set -euo pipefail

#################### 路径配置 ####################
# 你的仓库根目录（有 examples/video2world_lora.py 的地方）
REPO_ROOT="/scratch/10102/hh29499/carcrashtwin"

# 输入
FIRST_DIR="${REPO_ROOT}/raw_real41_img"
PROMPT_DIR="${REPO_ROOT}/txt_real41"
# 输出
OUT_ROOT="${REPO_ROOT}/cosmos_out_ft_344_10k_syn"
mkdir -p "$OUT_ROOT"

# 模型权重（按你给的命令来，走相对路径）
DIT_PATH="/scratch/10102/hh29499/carcrashtwin/checkpoints/posttraining/lora_10k/2b_custom_data_nuo_1019_10k/checkpoints/model/iter_000010000.pt"

#################### 分布式相关（slurm 传进来） ####################
WORLD_SIZE="${SLURM_NTASKS:-${WORLD_SIZE:-1}}"
WORKER_RANK="${SLURM_PROCID:-${WORKER_RANK:-0}}"
GPU_ID="${SLURM_LOCALID:-${GPU_ID:-0}}"

#################### 分布式相关 ####################

cd "$REPO_ROOT"

# 收集所有 prompt
mapfile -t PROMPTS < <(ls "$PROMPT_DIR"/*.txt | sort)
TOTAL=${#PROMPTS[@]}
echo "[INFO][rank ${WORKER_RANK}/${WORLD_SIZE}] total samples: $TOTAL"

for ((i=0; i<TOTAL; i++)); do
  # 轮询分配：谁的 rank 等于 i % WORLD_SIZE，谁来跑这个样本
  if (( i % WORLD_SIZE != WORKER_RANK )); then
    continue
  fi

  prompt_file="${PROMPTS[$i]}"
  base="$(basename "$prompt_file" .txt)"

  # 找图
  img_file="${FIRST_DIR}/${base}.png"
  if [ ! -f "$img_file" ]; then
    img_file="${FIRST_DIR}/${base}.jpg"
  fi
  if [ ! -f "$img_file" ]; then
    echo "[rank $WORKER_RANK] [WARN] no image for $base, skip."
    continue
  fi

  # 输出目录 & 输出文件
  outdir="${OUT_ROOT}/${base}"
  mkdir -p "$outdir"
  outmp4="${outdir}/${base}.mp4"

  # 已经有结果就跳过
  if [ -f "$outmp4" ]; then
    echo "[rank $WORKER_RANK] [SKIP] $base already done."
    continue
  fi

  echo "[rank $WORKER_RANK] run $base -> $outmp4"

  # ====== 真正的 cosmos LoRA 推理命令 ======
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  python examples/video2world_lora.py \
    --model_size 2B \
    --dit_path "$DIT_PATH" \
    --input_path "$img_file" \
    --prompt "$(head -n 1 "$prompt_file")" \
    --save_path "$outmp4" \
    --use_lora \
    --lora_rank 24 \
    --lora_alpha 24 \
    --lora_target_modules "q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2" \
    --disable_guardrail \
    --disable_prompt_refiner
  # ========================================

  echo "[rank $WORKER_RANK] done $base"
done

echo "[rank $WORKER_RANK] all assigned samples done."
