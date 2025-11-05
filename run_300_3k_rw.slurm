#!/bin/bash
#SBATCH -J cosmos_300_ft_3k_rw
#SBATCH -A CGAI24022
#SBATCH -p gh
#SBATCH -N 16
#SBATCH --ntasks-per-node=1
#SBATCH -t 10:00:00
#SBATCH -o /scratch/10102/hh29499/carcrashtwin/%x_%j.out
#SBATCH -e /scratch/10102/hh29499/carcrashtwin/%x_%j.err

srun --ntasks=${SLURM_JOB_NUM_NODES} --ntasks-per-node=1 --export=ALL \
  bash /scratch/10102/hh29499/carcrashtwin/run_300_ft_3krw.sh
