# MT5-small Fine-Tuning on FTRACE

This repository has been designed to provide a complete, reproducible workflow for fine-tuning the [google/mt5-small](https://huggingface.co/google/mt5-small) model on the [FTRACE dataset](https://huggingface.co/datasets/ekinakyurek/ftrace). The task has been formulated as **masked span prediction** using mT5’s `<extra_id_*>` tokens.

The entire training pipeline has been configured for execution on a **High-Performance Computing (HPC)** cluster using **Slurm** and **Singularity**, ensuring full reproducibility and seamless deployment.

---

## Key Features

- The `google/mt5-small` model has been fine-tuned on domain-specific abstracts (FTRACE).
- A **Singularity container** has been used to provide a reproducible environment.
- **Exact Match** and **ROUGE** metrics have been supported.
- A persistent Hugging Face **cache** has been implemented to accelerate subsequent runs.
- Logging has been facilitated through **TensorBoard**.
- A pre-configured **Slurm script** has been provided for cluster training.

---

## Folder Structure

```
mt5-finetune-ftrace/
├── cache/                  # Persistent cache for Hugging Face models
├── callbacks/              # Custom training callbacks
├── output/                 # All outputs (models, checkpoints, logs)
├── scripts/
│   ├── train_model.py      # The core model training script
│   ├── check_trained_model.py # Script to evaluate the trained model
│   └── prepare_dataset.py  # Script to preprocess and prepare the dataset
├── singularity/
│   ├── train.def           # Singularity container definition file
│   └── train.sif           # The built Singularity container
├── slurm/
│   └── fine_tune_mt5.sh    # SLURM script to run training on HPC
└── README.md
```

---

## Workflow Overview

```
Host Machine → SLURM Scheduler → Singularity Container → Python Training Script
```

- A Slurm job is submitted using `fine_tune_mt5.sh`.
- Compute resources are allocated by Slurm.
- The `train.sif` Singularity container is launched by the job.
- `train_model.py` is executed inside the container.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <https://github.com/denizakbulut1/mt5-small-finetune-ftrace.git>
cd mt5-finetune-ftrace
```

### 2. Prepare Dataset

It must be ensured that your preprocessed dataset is available at:

```
/data/datapool3/datasets/ftrace
```

If the dataset is located elsewhere, the path must be updated inside `slurm/fine_tune_mt5.sh`.

Alternatively, the dataset can be prepared using `scripts/prepare_dataset.py`.

### 3. Build Singularity Container

```bash
cd singularity/
singularity build --force train.sif train.def
cd ..
```

---

## Usage

### Run a Training Job (via Slurm)

```bash
sbatch ./slurm/fine_tune_mt5.sh
```

To track the job status:

```bash
squeue -u $USER
```

Slurm output logs will be stored in the `slurm/` directory.

---

### Interactive Debugging (Optional)

```bash
srun --pty --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=01:00:00 bash -i
```

Then:

```bash
cd /path/to/mt5-finetune-ftrace
bash ./slurm/submit_training.sh
```

---

## Project Outputs

- **Trained Model**: Saved to `output/mt5-finetuned-ftrace/`
- **TensorBoard Logs**: Stored within the same output folder
- **Hugging Face Cache**: Located in `cache/`

---

## Customization

- **Training Parameters**: Can be changed in `scripts/train_model.py`
- **Job Configuration**: Can be updated in `slurm/fine_tune_mt5.sh`
- **Python Environment**: Can be modified in `singularity/train.def` (requires container rebuild)


## Example Bind Command (Manual Run)

```bash
singularity run \
  --bind /data/datapool3/datasets/ftrace:/mnt/train_data \
  --bind /home/fe/akbulut/mt5-finetune-ftrace/output:/mnt/outputs \
  --bind /home/fe/akbulut/mt5-finetune-ftrace/cache:/tmp/huggingface \
  singularity/train.sif
```

---

## `Seq2SeqTrainingArguments` Options (with Alternatives)

```python
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="/mnt/outputs/mt5-finetuned-ftrace",  # Path where model checkpoints will be written
    logging_dir="/mnt/outputs/mt5-finetuned-ftrace/logs",  # Directory for TensorBoard logs
    disable_tqdm=True,  # Alternatives: True (disable tqdm) / False (show progress bar)
    evaluation_strategy="no",  # Alternatives: "no", "steps", "epoch"
    save_strategy="epoch",  # Alternatives: "no", "steps", "epoch"
    logging_strategy="steps",  # Alternatives: "no", "epoch", "steps"
    logging_steps=50000,  # Logging frequency (only relevant if strategy is "steps")
    per_device_train_batch_size=8,  # Alternatives: any int (e.g. 4, 8, 16, 32) depending on GPU memory
    per_device_eval_batch_size=32,
    num_train_epochs=50,  # Number of epochs to train
    learning_rate=3e-5,  # Alternatives: 5e-5, 3e-4, etc. (tune as needed)
    save_total_limit=1,  # Max number of checkpoints to save
    predict_with_generate=True,  # If True, generation is used during prediction
    fp16=False,  # Alternatives: True (use FP16 on compatible GPUs) / False
    report_to="tensorboard",  # Alternatives: "tensorboard", "wandb", "none"
    load_best_model_at_end=False,  # Alternatives: True (requires eval + metric) / False
    metric_for_best_model="exact_match",  # Optional, only needed if load_best_model_at_end=True
    greater_is_better=True  # Optional, relevant for metric direction
)
```
