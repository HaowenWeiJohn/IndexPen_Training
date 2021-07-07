#!/bin/bash
#SBATCH --job-name="12-2020_all_data_stratify_clutter_removal_0.8_0.6_ratio_test_without_minimax_complex_model_zero_thresholding"
#SBATCH -D .
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --partition=long
#SBATCH --time=150:00:00


module purge
module load gcc/6.3.0 slurm/17.02.11 cudnn/8.1.1.33-11.2/3k5bbs63 cuda/11.0.2/ur4svoci 


python training_script.py