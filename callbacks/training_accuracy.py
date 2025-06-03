from transformers import TrainerCallback
import torch
from torch.utils.tensorboard import SummaryWriter
import os

class TrainingAccuracyCallback(TrainerCallback):
    def __init__(self, tokenizer, dataset, metric, print_every=50000, log_dir="./logs"):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.metric = metric
        self.print_every = print_every
        self.counter = 0

        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def on_step_end(self, args, state, control, **kwargs):
        self.counter += 1
        if self.counter % self.print_every == 0:
            model = kwargs["model"]
            model.eval()
            sample = self.dataset.shuffle(seed=state.global_step).select(range(8))

            inputs = self.tokenizer(
                sample["inputs_pretokenized"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(**inputs, max_length=64)

            decoded_preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            decoded_labels = sample["targets_pretokenized"]

            result = self.metric.compute(predictions=decoded_preds, references=decoded_labels)
            acc = round(result["exact_match"], 4)

            # Log to console
            print(f"\nStep {state.global_step} — Training Exact Match: {acc}")

            # Log to TensorBoard
            self.writer.add_scalar("Training/Exact_Match", acc, state.global_step)

            model.train()

        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.writer.close()
