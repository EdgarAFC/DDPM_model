#!/usr/bin/bash
#SBATCH --gpus-per-node=1
#SBATCh --nodes=1
#SBATCH --partition=gamerpcs
#SBATCH --nodelist=worker2
#SBATCH --output="log_model.out"

srun python /nfs/privileged/edgar/projects/DPM_model/main.py
