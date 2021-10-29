#!/bin/bash
#SBATCH --job-name="study2_run_all_3_test"
#SBATCH -D .
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

#SBATCH --partition=long
#SBATCH --time=150:00:00


module purge
module load gcc/6.3.0 slurm/17.02.11 cudnn/8.1.1.33-11.2/3k5bbs63 cuda/11.2.2/x7rjvfkn


python participant_run_all_3.py
