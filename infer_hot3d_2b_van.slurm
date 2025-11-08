#!/bin/bash
#SBATCH -J cosmos_hot3d_van_2b
#SBATCH -A CGAI24022
#SBATCH -p gh
#SBATCH -N 16
#SBATCH --ntasks-per-node=1
#SBATCH -t 10:00:00
#SBATCH -o /scratch/10102/hh29499/carcrashtwin/%x_%j.out
#SBATCH -e /scratch/10102/hh29499/carcrashtwin/%x_%j.err

srun  --export=ALL \
  bash /scratch/10102/hh29499/carcrashtwin/infer_hot3d_2b_van.sh