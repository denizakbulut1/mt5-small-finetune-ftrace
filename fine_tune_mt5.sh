#!/bin/bash
#SBATCH --job-name=mt5_finetune
#SBATCH --output=/home/fe/akbulut/mt5-finetune-ftrace/output/job_output.log
#SBATCH --error=/home/fe/akbulut/mt5-finetune-ftrace/output/job_error.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=deniz.akbulut@hhi.fraunhofer.de

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mt5cluster

# Avoid memory fragmentation + suppress warnings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KMP_DUPLICATE_LIB_OK=True
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# Navigate to project folder
cd /home/fe/akbulut/mt5-finetune-ftrace

#Train the model

echo "Starting training..."
python scripts/train_model.py

#Evaluate the model

echo "Starting evaluation..."
python scripts/evaluate_model.py

echo "Training and evaluation finished."
