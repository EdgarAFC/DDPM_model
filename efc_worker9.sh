#!/usr/bin/bash
#SBATCH --gpus-per-node=1
#SBATCh --nodes=1
#SBATCH --partition=thinkstation-p360
#SBATCH --nodelist=worker9
#SBATCH --output="log_model_sig.out"

source /etc/profile.d/modules.sh
module load ifsr-advpertbeamf/1.0
srun python /mnt/nfs/efernandez/projects/DDPM_model/main.py
