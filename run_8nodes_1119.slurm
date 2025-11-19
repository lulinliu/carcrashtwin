#!/bin/bash
#SBATCH -J cosmos_1119
#SBATCH -A CGAI24022
#SBATCH -p gh
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 40:00:00
#SBATCH -o /scratch/10102/hh29499/carcrashtwin/%x_%j.out
#SBATCH -e /scratch/10102/hh29499/carcrashtwin/%x_%j.err

srun --ntasks=${SLURM_JOB_NUM_NODES} --ntasks-per-node=1 --export=ALL \
  bash /scratch/10102/hh29499/carcrashtwin/singleone_1119.sh
