#!/bin/bash
#SBATCH -J cosmos_8n1g
#SBATCH -A CGAI24022    
#SBATCH --partition=gh
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 16:00:00
#SBATCH -o /scratch/10102/hh29499/carcrashtwin/%x_%j.out
#SBATCH -e /scratch/10102/hh29499/carcrashtwin/%x_%j.err


set -euo pipefail

#################### 可按需修改的变量 ####################
# 两个 manifest：各有 10 行，逐行给出“每个节点要用的 txt 列表文件”的路径
VIDEOS_MANIFEST="tasks_videos_manifest.txt"
PROMPTS_MANIFEST="tasks_prompts_manifest.txt"
export PROJECT_DIR="/scratch/10102/hh29499/carcrashtwin"          # repo root where examples.video2world lives
export VIDEOS_MANIFEST="/scratch/10102/hh29499/carcrashtwin/tasks_videos_manifest.txt"
export PROMPTS_MANIFEST="/scratch/10102/hh29499/carcrashtwin/tasks_prompts_manifest.txt"
export OUTPUT_ROOT="/scratch/10102/hh29499/carcrashtwin/output_video2world_$(date +%Y%m%d_%H%M%S)"

# 每个节点对应的列表 txt 里：逐行给出单个视频路径 / 单个 prompt，一一对应
export OUTPUT_ROOT="/scratch/10102/hh29499/carcrashtwin/output_video2world_$(date +%Y%m%d_%H%M%S)"

export MODEL_SIZE="2B"
export COND_FRAMES="1"

mkdir -p slurm_logs
mkdir -p "${OUTPUT_ROOT}"

# 每个任务各跑一个 worker；worker 会用 SLURM_PROCID 选取自己的那一行
srun --ntasks=${SLURM_NTASKS} bash ./run_10uq.sh
echo "[INFO] Job ends at: $(date)"
