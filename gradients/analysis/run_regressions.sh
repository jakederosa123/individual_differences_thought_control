#!/bin/bash

#SBATCH --job-name=regressions
#SBATCH --partition=amilan
##SBATCH --cpus-per-task=24 
#SBATCH --mem-per-cpu=3500   # You may adjust this if more memory is needed per CPU
#SBATCH --time=24:00:00      # Specifies the maximum time for the job

python3 /pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/analysis/regressions.py
