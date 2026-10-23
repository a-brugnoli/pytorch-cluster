#!/bin/bash
#SBATCH --job-name=torch-benchmark
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=benchmark.%j.out
#SBATCH --error=benchmark.%j.err

set -e
set -o pipefail

# Use the same module as your cluster's verification script.
module load python/2023-3
eval "$(conda shell.bash hook)"
conda activate pytorch-wheel

# Submit from the directory containing test_cpu_gpu.py.
cd "$SLURM_SUBMIT_DIR"
test -f test_cpu_gpu.py || {
    echo "Missing test_cpu_gpu.py in $PWD" >&2
    exit 1
}

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

printf 'Job: %s | Node: %s | Start: %s\n' \
    "$SLURM_JOB_ID" "$(hostname)" "$(date -Iseconds)"
python -c 'import sys; print("Python:", sys.executable)'
nvidia-smi

# Compare CPU and CUDA on the same allocated node.
# The benchmark performs warmup, synchronizes GPU timing, and reports medians.
# Arguments passed after the sbatch filename can override the benchmark defaults.
srun python -u test_cpu_gpu.py \
    --require-cuda \
    --samples 4096 \
    --width 128 \
    --steps 300 \
    --warmup 20 \
    --repeats 3 \
    --threads "$SLURM_CPUS_PER_TASK" \
    --output "results/benchmark-${SLURM_JOB_ID}.json" \
    "$@"

printf 'Finished: %s\n' "$(date -Iseconds)"
