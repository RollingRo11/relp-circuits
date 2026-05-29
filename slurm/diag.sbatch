#!/bin/bash
#SBATCH --job-name=relp-diag
#SBATCH --partition=dev
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=/data/artifacts/rohan/relp-circuits/logs/diag-%j.out
#SBATCH --error=/data/artifacts/rohan/relp-circuits/logs/diag-%j.err

set -euo pipefail

echo "=== hostname ==="; hostname
echo "=== CUDA_VISIBLE_DEVICES ==="; echo "${CUDA_VISIBLE_DEVICES:-unset}"
echo "=== slurm gres ==="; echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "=== nvidia-smi -L (visible GPUs in this slurm allocation) ==="; nvidia-smi -L
echo "=== nvidia-smi memory + processes for VISIBLE GPUs ==="; nvidia-smi
echo "=== whoami ==="; whoami
echo "=== ps for vLLM PIDs 2546335..2546342 ==="
ps -o pid,user,comm,etime,args -p 2546335,2546336,2546337,2546338,2546339,2546340,2546341,2546342 2>&1 | head -20
echo "=== my processes on this node ==="
ps -u rkathuria -o pid,user,etime,comm 2>&1 | head -20
echo "=== other slurm jobs on this node ==="
squeue -w "$(hostname)" 2>&1 | head
