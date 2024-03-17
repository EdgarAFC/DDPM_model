#!/usr/bin/bash
#SBATCH --gpus-per-node=1
#SBATCh --nodes=1
#SBATCH --partition=gamerpcs
#SBATCH --nodelist=worker2
#SBATCH --output="log_model.out"

source /etc/profile.d/modules.sh
module load edgar/1.0
srun python /nfs/privileged/edgar/projects/DDPM_model/split_fold.py
