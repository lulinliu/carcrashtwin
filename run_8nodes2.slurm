#!/bin/bash
#SBATCH -J cosmos_hot3d_14
#SBATCH -A CGAI24022
#SBATCH -p gh
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 60:00:00
#SBATCH -o /scratch/10102/hh29499/carcrashtwin/%x_%j.out
#SBATCH -e /scratch/10102/hh29499/carcrashtwin/%x_%j.err

srun --ntasks=${SLURM_JOB_NUM_NODES} --ntasks-per-node=1 --export=ALL \
  bash /scratch/10102/hh29499/carcrashtwin/run1.sh
