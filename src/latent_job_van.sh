#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# SBATCH --nodelist=nodo01,nodo02,nodo04,nodo07,nodo08,nodo09,nodo10,nodo11,nodo12
#SBATCH --exclude=nodo17,nodo13,nodo15,nodo20,nodo18,nodo19,nodo21,nodo22,nodo23,nodo24
#SBATCH --workdir=/home/emiliano/latentnoise_krr/src/
#SBATCH --job-name=latentnoise_krr_van
#SBATCH --output=/home/emiliano/latentnoise_krr/log/latentNoise_van_%A_%a.out
#SBATCH --array=1-6012%100

# initialize conda environment on ERC
module load Anaconda3
# activate environment or your script!
source activate lnkrr_py38

# DO STUFF
srun --ntasks 1 --nodes 1  python -u slurm_script_van.py --save v0 --job $SLURM_ARRAY_TASK_ID --offset 0 --server erc
