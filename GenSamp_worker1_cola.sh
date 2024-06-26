#!/usr/bin/bash
#SBATCH --gpus-per-node=1
#SBATCh --nodes=1
#SBATCH --partition=gamerpcs
#SBATCH --nodelist=worker2
#SBATCH --output="log_model_w2.out"

source /etc/profile.d/modules.sh
module load ifsr-advpertbeamf/1.0
srun python /mnt/nfs/efernandez/projects/DDPM_model/gen_samples_1_cola.py
