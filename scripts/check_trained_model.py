import torch
import os
import evaluate
import json
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from datasets import load_from_disk
from transformers import (
    MT5Tokenizer,
    MT5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

FINETUNED_MODEL_PATH = "/mnt/outputs/mt5-finetuned-ftrace"
DATASET_PATH = "/mnt/train_data"
EVAL_OUTPUT_DIR = "/mnt/outputs/tmp-eval"

SUMMARY_FILE_PATH = os.path.join(FINETUNED_MODEL_PATH, "evaluation_summary.txt")

print("--- Setting up environment ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print(f"Loading fine-tuned model from: {FINETUNED_MODEL_PATH}")
tokenizer = MT5Tokenizer.from_pretrained(FINETUNED_MODEL_PATH, use_fast=False)
finetuned_model = MT5ForConditionalGeneration.from_pretrained(FINETUNED_MODEL_PATH).to(device).eval()

print("Loading base 'google/mt5-small' model...")
base_model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small").to(device).eval()

print(f"Loading dataset from: {DATASET_PATH}")
dataset = load_from_disk(DATASET_PATH)
eval_dataset = dataset["train"].shuffle(seed=42).select(range(200, 250))
print(f"Loaded {len(eval_dataset)} samples for evaluation.")

exact_match_metric = evaluate.load("exact_match")
rouge_metric = evaluate.load("rouge")

# Test 1: Single Inference Example
print("\n--- Running single inference test ---")
input_text = ("Sodium and potassium are also essential elements, having major biological roles as electrolytes, and although the other <extra_id_0> are not essential, they also "
"have various effects on the body, both beneficial and harmful.")
inputs = tokenizer(input_text, return_tensors="pt").to(device)
output_ids = finetuned_model.generate(**inputs, max_length=50, num_beams=5)
decoded_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"Input Text: {input_text}")
print(f"Predicted Output: {decoded_output}")

# Test 2: Formal Evaluation with Seq2SeqTrainer
print("\n--- Running formal evaluation with Trainer.evaluate() ---")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    rouge_result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    em_result = exact_match_metric.compute(predictions=decoded_preds, references=decoded_labels)
    rouge_result["exact_match"] = round(em_result["exact_match"], 4)
    
    return {k: round(v, 4) for k, v in rouge_result.items()}

trainer = Seq2SeqTrainer(
    model=finetuned_model,
    args=Seq2SeqTrainingArguments(
        output_dir=EVAL_OUTPUT_DIR,
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        report_to="none"
    ),
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=finetuned_model),
    compute_metrics=compute_metrics,
)
trainer_metrics = trainer.evaluate()
print("Evaluation Metrics from Trainer:", trainer_metrics)

# Test 3: Before vs. After Comparison
print("\n--- Comparing performance: Before vs. After Fine-Tuning ---")
input_texts = [tokenizer.decode(example["input_ids"], skip_special_tokens=True) for example in eval_dataset]
references = [tokenizer.decode(example["labels"], skip_special_tokens=True) for example in eval_dataset]

def get_exact_match(model, texts):
    predictions = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=128)
        predictions.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
    
    result = exact_match_metric.compute(predictions=predictions, references=references)
    return round(result["exact_match"], 4)

em_base = get_exact_match(base_model, input_texts)
em_finetuned = get_exact_match(finetuned_model, input_texts)

print(f"Exact Match Score BEFORE training (base model): {em_base}")
print(f"Exact Match Score AFTER fine-tuning: {em_finetuned}")

# Save Results to Summary File
print(f"\n--- Saving results to {SUMMARY_FILE_PATH} ---")
with open(SUMMARY_FILE_PATH, 'w') as f:
    f.write("Evaluation Summary\n")
    f.write("==================\n")
    f.write(f"Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Model Path: {FINETUNED_MODEL_PATH}\n")
    f.write(f"Dataset Path: {DATASET_PATH}\n")
    f.write(f"Evaluation Samples: {len(eval_dataset)}\n\n")

    f.write("Trainer Evaluation Metrics:\n")
    
    f.write(json.dumps(trainer_metrics, indent=2))
    f.write("\n\n")
    
    f.write("Before vs. After Comparison:\n")
    f.write(f"  Exact Match Score (Base Model): {em_base}\n")
    f.write(f"  Exact Match Score (Fine-Tuned Model): {em_finetuned}\n")

print("Summary file saved successfully.")