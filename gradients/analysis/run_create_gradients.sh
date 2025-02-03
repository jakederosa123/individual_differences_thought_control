#!/bin/bash

#SBATCH --job-name=gradients
#SBATCH --account=ucb-general
#SBATCH --partition=ami100
#SBATCH --nodes=4
#SBATCH --ntasks=24
#SBATCH --tasks-per-node=6
#SBATCH --gres=gpu:1
#SBATCH --mem=50G



source /curc/sw/anaconda3/latest

conda activate jake

python3 /pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/analysis/create_ind_gradients_new.py


#Terminal: 
#cd /pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/
#nohup sbatch run_gradients

#1. Does nohup work with sbatch
#2. If nohup doesnt work, does it work with srun
#3. sinteractive session 