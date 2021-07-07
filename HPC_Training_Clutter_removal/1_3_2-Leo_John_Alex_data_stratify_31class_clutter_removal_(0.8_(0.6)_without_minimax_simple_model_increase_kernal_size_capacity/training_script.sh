#!/bin/bash
#SBATCH --job-name="1_3_2-Leo_John_Alex_data_stratify_31class_clutter_removal_(0.8_(0.6)_without_minimax_simple_model_increase_kernal_size_capacity"
#SBATCH -D .
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --partition=long
#SBATCH --time=100:00:00


module purge
module load gcc/6.3.0 slurm/17.02.11 cudnn/8.1.1.33-11.2/3k5bbs63 cuda/11.0.2/ur4svoci 


python training_script.py