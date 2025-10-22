#!/bin/bash
#SBATCH -J dummy_8n
#SBATCH -A CGAI24022
#SBATCH -p gh
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:05:00
#SBATCH -o /scratch/10102/hh29499/carcrashtwin/%x_%j.out
#SBATCH -e /scratch/10102/hh29499/carcrashtwin/%x_%j.err

module purge; module load gcc/13.2.0; module load cuda/12.6
srun --ntasks=${SLURM_JOB_NUM_NODES} --ntasks-per-node=1 --export=ALL \
  bash /scratch/10102/hh29499/carcrashtwin/dum.sh
