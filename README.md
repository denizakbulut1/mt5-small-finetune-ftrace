# MT5-small Fine-Tuning on FTRACE

This repository contains code and scripts for fine-tuning the [mT5-small](https://huggingface.co/google/mt5-small) model on the [FTRACE dataset](https://huggingface.co/datasets/ekinakyurek/ftrace). The task involves masked span prediction using mT5’s `<extra_id_*>` tokens.

## Project

- Fine-tunes `google/mt5-small` on domain-specific abstracts
- Uses Hugging Face `Seq2SeqTrainer` with custom metrics
- Logs training via TensorBoard
- Supports exact match and ROUGE evaluations

## Folder Structure

```
mt5-small-finetune-ftrace/
├── callbacks/              # Custom training callback
├── scripts/                # Training, evaluation, and dataset prep scripts
├── fine_tune_mt5.sh        # SLURM script for training on a cluster
├── mt5-small.yml           # Conda environment file
├── README.md               # This file
└── .gitignore              # Files/folders excluded from GitHub
```

