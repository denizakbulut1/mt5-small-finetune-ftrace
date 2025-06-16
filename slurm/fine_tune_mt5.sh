#!/bin/bash
#SBATCH --job-name=mt5_finetune_singularity
#SBATCH --output=/home/fe/akbulut/mt5-finetune-ftrace/slurm/job_output_%j.log
#SBATCH --error=/home/fe/akbulut/mt5-finetune-ftrace/slurm/job_error_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=deniz.akbulut@hhi.fraunhofer.de

# --- Environment Setup ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mt5cluster

# Set environment variables for PyTorch and Transformers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# --- Execution ---
echo "Changing directory to project root..."
cd /home/fe/akbulut/mt5-finetune-ftrace
if [ $? -ne 0 ]; then
  echo "Failed to cd to project root. Exiting."
  exit 1
fi

echo "Current working directory: $(pwd)"

# Define paths for clarity
SIF_PATH="./singularity/train.sif"
DATA_PATH="/data/datapool3/datasets/ftrace"
OUTPUT_PATH="./output"
CACHE_PATH="./cache"

# Check if the SIF file exists
if [ ! -f "$SIF_PATH" ]; then
    echo "Singularity image not found at $SIF_PATH. Exiting."
    exit 1
fi

# --- Train the model using Singularity ---
echo "Starting training inside Singularity container..."
singularity run --nv \
  --bind ${DATA_PATH}:/mnt/train_data \
  --bind ${OUTPUT_PATH}:/mnt/outputs \
  --bind ${CACHE_PATH}:/mnt/cache \
  "$SIF_PATH"

echo "Training finished."

# --- Evaluate the model using Singularity ---
echo "Starting evaluation inside Singularity container..."
singularity exec --nv \
  --bind ${DATA_PATH}:/mnt/train_data \
  --bind ${OUTPUT_PATH}:/mnt/outputs \
  --bind ${CACHE_PATH}:/mnt/cache \
  "$SIF_PATH" \
  python /opt/scripts/check_trained_model.py

echo "Evaluation finished. Job complete."