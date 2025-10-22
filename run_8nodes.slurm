#!/bin/bash
#SBATCH -J cosmos_lora_8n
#SBATCH -A CGAI24022
#SBATCH -p gh
#SBATCH -N 8
#SBATCH -t 16:00:00
#SBATCH -o /scratch/10102/hh29499/carcrashtwin/%x_%j.out
#SBATCH -e /scratch/10102/hh29499/carcrashtwin/%x_%j.err
#SBATCH --ntasks-per-node=1        # one task per node (1 GPU per node). DO NOT use --gpus-per-node/--gres

set -euo pipefail

module purge
module load gcc/13.2.0
module load cuda/12.6

PROJECT_DIR=/scratch/10102/hh29499/carcrashtwin
cd "$PROJECT_DIR"
source "$(dirname $(dirname $(which conda)))/etc/profile.d/conda.sh"
conda init
conda activate cosmos-predict2

# NCCL/CPU env (safe defaults)
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=^lo,docker0
export OMP_NUM_THREADS=8

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=${MASTER_PORT:-12341}
NNODES=${SLURM_JOB_NUM_NODES}
NPROC_PER_NODE=1

# Launch one task per node; each task runs one torchrun worker with its NODE_RANK.
srun --ntasks=${NNODES} --ntasks-per-node=1 --kill-on-bad-exit=1 bash -lc '
  export NODE_RANK=${SLURM_PROCID}
  torchrun \
    --nproc_per_node='"${NPROC_PER_NODE}"' \
    --nnodes='"${NNODES}"' \
    --node_rank=${NODE_RANK} \
    --master_addr='"${MASTER_ADDR}"' \
    --master_port='"${MASTER_PORT}"' \
    -m scripts.train \
    --config=cosmos_predict2/configs/base/config.py \
    -- experiment=predict2_video2world_lora_training_2b_1019nuo_full
'
