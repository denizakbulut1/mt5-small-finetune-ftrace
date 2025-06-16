import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from datasets import load_from_disk
from transformers import (
    MT5Tokenizer,
    MT5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)

from callbacks.training_accuracy import TrainingAccuracyCallback
import evaluate
from datasets import load_dataset

dataset = load_from_disk("/mnt/train_data")

train_dataset = dataset["train"]  # Use all abstracts
eval_dataset = None

# Load raw dataset for accuracy callback
raw_dataset = train_dataset

tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small", use_fast=False)
model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")

# Data collator to handle padding
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

exact_match = evaluate.load("exact_match")

"""

# Load metrics
rouge = evaluate.load("rouge")
exact_match_metric = evaluate.load("exact_match")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Decode predicted and reference token IDs
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute ROUGE
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels)

    # Compute Exact Match using evaluate
    em_result = exact_match_metric.compute(predictions=decoded_preds, references=decoded_labels)
    exact = round(em_result["exact_match"], 4)
    result["exact_match"] = exact

    # Optional: log to console
    print(f"🔍 Exact Match: {exact}")

    # Return rounded metrics
    return {k: round(v, 4) for k, v in result.items()}
"""

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="/mnt/outputs/mt5-finetuned-ftrace",
    logging_dir="/mnt/outputs/mt5-finetuned-ftrace/logs",
    disable_tqdm=True,
    evaluation_strategy="no",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50000,
    per_device_train_batch_size=8,
    #per_device_eval_batch_size=32,
    num_train_epochs=50,
    learning_rate=3e-5,
    save_total_limit=1,
    predict_with_generate=True,
    fp16=False,
    report_to="tensorboard",
    load_best_model_at_end=False,
    #metric_for_best_model="exact_match",
    #greater_is_better=True
)


print(f"Dataset size: {len(train_dataset)}")
print(f"Batch size: {training_args.per_device_train_batch_size}")
print(f"Estimated steps/epoch: {len(train_dataset) // training_args.per_device_train_batch_size}")


# Trainer with early stopping callback
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    #eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    #compute_metrics=compute_metrics,
    callbacks=[
        TrainingAccuracyCallback(
            tokenizer=tokenizer,
            dataset=raw_dataset,
            metric=exact_match,
            print_every=50000
            )
            ]
)

# Start training
trainer.train()

# Save the final model
trainer.save_model("/mnt/outputs/mt5-finetuned-ftrace")
print("Training complete and model saved.")
