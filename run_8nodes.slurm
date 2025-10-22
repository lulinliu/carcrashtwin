#!/bin/bash
#SBATCH -J cosmos_lora_8g
#SBATCH -A CGAI24022
#SBATCH --ntasks-per-node=1       # 每节点 1 个进程
#SBATCH --gpus-per-node=1 
#SBATCH -o /scratch/10102/hh29499/carcrashtwin/%x_%j.out
#SBATCH -e /scratch/10102/hh29499/carcrashtwin/%x_%j.err


# --------- 路径与模块 ---------
module purge
module load cuda/12.6               # 你们用的 12.4
module load miniconda3               # TACC 提供的 conda
# 如果没有 logs 目录，建一个


# 项目工作目录（按你的实际目录修改）
PROJECT_DIR=/scratch/10102/hh29499/carcrashtwin
cd "$PROJECT_DIR"

# 激活你的 conda 环境
conda activate cosmos-predict2

# --------- 通用/分布式环境变量（稳定训练常用） ---------
# 避免 NCCL 静默挂起、便于排错
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
# 常见网络接口排除回环
export NCCL_SOCKET_IFNAME=^lo,docker0
# 线性代数/CPU 线程控制（按需调整）
export OMP_NUM_THREADS=8

# --------- 以 srun 启动 8 进程（每进程 1 卡）---------
# 你的训练命令保持不变；srun 会为每个 task 设定 CUDA_VISIBLE_DEVICES
srun --kill-on-bad-exit=1 \
  torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=predict2_video2world_lora_training_2b_1019nuo_full
