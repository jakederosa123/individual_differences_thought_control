#!/bin/bash

#SBATCH --job-name=abcd
#SBATCH --partition=amilan
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=12
##SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=3500

python3 /pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/analysis/gradient_network_plots.py