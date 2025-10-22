#!/bin/bash
#SBATCH -J cosmos_lora_8n
#SBATCH -A CGAI24022
#SBATCH -p gh
#SBATCH -N 8
#SBATCH -t 16:00:00
#SBATCH -o /scratch/10102/hh29499/carcrashtwin/%x_%j.out
#SBATCH -e /scratch/10102/hh29499/carcrashtwin/%x_%j.err
#SBATCH --ntasks-per-node=1

set -euo pipefail

module purge
module load gcc/13.2.0
module load cuda/12.6

PROJECT_DIR=/scratch/10102/hh29499/carcrashtwin
cd "$PROJECT_DIR"

# caches (optional)
export HF_HOME="${HF_HOME:-$SCRATCH/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export TORCH_HOME="${TORCH_HOME:-$HF_HOME/torch}"

# NCCL/CPU env
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=${MASTER_PORT:-12341}
NNODES=${SLURM_JOB_NUM_NODES}

srun --ntasks=${NNODES} --ntasks-per-node=1 --export=ALL --kill-on-bad-exit=1 bash -lc '
  module load gcc/13.2.0
  module load cuda/12.6

  # conda
  set +u
  source /scratch/10102/hh29499/anaconda/etc/profile.d/conda.sh
  conda activate cosmos-predict2
  set -u

  # 👇 kill the shadowing
  unset PYTHONPATH

  # quick sanity: ensure we get HF tokenizers from site-packages
  python - << "PY"
import tokenizers, sys, os
print("tokenizers from:", tokenizers.__file__)
assert "site-packages/tokenizers" in tokenizers.__file__, "Still shadowed by local package!"
PY

  export NODE_RANK=${SLURM_PROCID}

  # run from repo root so local packages (cosmos_predict2, imaginaire, scripts) are importable
  cd /scratch/10102/hh29499/carcrashtwin

  torchrun \
    --nproc_per_node=1 \
    --nnodes='"${NNODES}"' \
    --node_rank=${NODE_RANK} \
    --master_addr='"${MASTER_ADDR}"' \
    --master_port='"${MASTER_PORT}"' \
    -m scripts.train \
    --config=cosmos_predict2/configs/base/config.py \
    -- experiment=predict2_video2world_lora_training_2b_1019nuo_full
'
