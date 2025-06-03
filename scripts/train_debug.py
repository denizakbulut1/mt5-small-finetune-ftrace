import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from datasets import load_from_disk
from transformers import (
    MT5Tokenizer,
    MT5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
import evaluate

from transformers import Seq2SeqTrainingArguments
print("Loaded from:", Seq2SeqTrainingArguments.__module__)


# Load tokenized dataset
dataset = load_from_disk("data/tokenized_abstracts")

# Use a small subset for debugging
shuffled = dataset["train"].shuffle(seed=42)
train_dataset = shuffled.select(range(200))
eval_dataset = shuffled.select(range(200, 250))

# Load tokenizer and model
tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small", use_fast=False)
model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")

# Data collator for padding
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

"""
rouge = evaluate.load("rouge")
exact_match_metric = evaluate.load("exact_match")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    exact_match = sum(
        int(pred.strip() == label.strip())
        for pred, label in zip(decoded_preds, decoded_labels)
    ) / len(decoded_preds)
    result["exact_match"] = exact_match

    print(f"🔍 Exact Match: {round(exact_match, 4)}")
    return {k: round(v, 4) for k, v in result.items()
    }
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



# Debug training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./outputs/debug-mt5",
    logging_dir="./logs/debug",
    evaluation_strategy="epoch",
    save_strategy="no",
    logging_strategy="steps",
    logging_steps=10,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=1,
    learning_rate=3e-5,
    predict_with_generate=True,
    fp16=False,
    report_to="none",
    load_best_model_at_end=False
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


trainer.train()

print("Debug training finished successfully!")
