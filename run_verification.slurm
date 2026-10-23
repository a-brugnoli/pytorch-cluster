#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --job-name=my_program
#SBATCH --output=slurm.%j.out
#SBATCH --error=slurm.%j.err

#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

set -e

# If necessary, load your cluster's Conda module here.
module load python/2023-3

eval "$(conda shell.bash hook)"
conda activate pytorch-env

cd "$SLURM_SUBMIT_DIR"

printf 'Job: %s | Node: %s | Start: %s\n' \
    "$SLURM_JOB_ID" "$(hostname)" "$(date -Iseconds)"

python -c 'import sys; print("Python:", sys.executable)'
bash ./verify_torch_installation.sh