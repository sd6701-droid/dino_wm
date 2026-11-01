#!/bin/bash
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --job-name=plan_pusht
#SBATCH --output=/gpfs/scratch/sd6701/personal/salt/dino_wm/logs/%x_%j.log
#SBATCH --error=/gpfs/scratch/sd6701/personal/salt/dino_wm/logs/%x_%j.err

# Evaluates the shared PushT checkpoint with CEM inside MPC (conf/plan_pusht.yaml), 1xA100.
# The checkpoint has no encoder, so plan.py loads dinov2_vits14 from torch.hub.
# sbatch needs the logs folder to exist before the job starts: mkdir -p logs
#
# Test (5 tasks):
#   sbatch --partition=a100_dev --time=00:30:00 --export=ALL,N_EVALS=5 \
#     sbatch/bigpurple/plan_pusht.sbatch
#
# Full evaluation (50 tasks):
#   sbatch sbatch/bigpurple/plan_pusht.sbatch

set -euo pipefail

cd /gpfs/scratch/sd6701/personal/salt/dino_wm

eval "$(conda shell.bash hook)"
conda activate /gpfs/scratch/sd6701/conda-envs/dino_wm

export PYTHONNOUSERSITE=1
# Home is over its file quota: keep caches and W&B files on scratch.
export XDG_CACHE_HOME=/gpfs/scratch/sd6701/.cache
export TORCH_HOME=/gpfs/scratch/sd6701/.cache/torch
export WANDB_DIR=/gpfs/scratch/sd6701/wandb
mkdir -p "$WANDB_DIR"
export DATASET_DIR=/gpfs/scratch/sd6701/personal/salt/datasets/dinowm_dataset
export HYDRA_FULL_ERROR=1

python plan.py --config-name plan_pusht.yaml \
  ckpt_base_path=/gpfs/scratch/sd6701/dino_wm_ckpts \
  model_name=pusht \
  n_evals="${N_EVALS:-50}"
