#!/bin/bash
#SBATCH --job-name="13-2020_all_data_clutter_removal_0.8_0.6_ratio_test_without_minimax_simple_model_unit_sample_minimax"
#SBATCH -D .
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:K80:1
#SBATCH --partition=long
#SBATCH --time=60:00:00


module purge
module load gcc/6.3.0 slurm/17.02.11 cudnn/8.1.1.33-11.2/3k5bbs63 cuda/11.0.2/ur4svoci 


python training_script.py