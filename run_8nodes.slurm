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

PROJECT_DIR=/scratch/10102/hh29499/carcrashtwin
cd "$PROJECT_DIR"


MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=${MASTER_PORT:-12341}
NNODES=${SLURM_JOB_NUM_NODES}

srun --ntasks=${NNODES} --ntasks-per-node=1  bash -lc '
  # avoid local shadowing
  export NODE_RANK=${SLURM_PROCID}
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
