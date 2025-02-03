#!/bin/bash

#SBATCH --job-name=abcd
#SBATCH --partition=amilan
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=24
##SBATCH --gres=gpu
#SBATCH --mem-per-cpu=3500

python3 /pl/active/banich/studies/wmem/fmri/operation_rsa/grp/gradients/analysis/subject_network_distances.py
