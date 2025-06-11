import torch

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


# Load model and tokenizer
model_path = "./outputs/mt5-finetuned-ftrace/checkpoint-2925855"
tokenizer = MT5Tokenizer.from_pretrained(model_path, use_fast=False)
model = MT5ForConditionalGeneration.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.eval()

input_text = ("Sodium and potassium are also essential elements, having major biological roles as electrolytes, and although the other <extra_id_0> are not essential, they also "
"have various effects on the body, both beneficial and harmful.")

inputs = tokenizer(input_text, return_tensors="pt").to(device)
output_ids = model.generate(**inputs, max_length=50, num_beams=5)
decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Predicted span:", decoded)

# Remove extra tokens
if "<extra_id_0>" in decoded and "<extra_id_1>" in decoded:
    span = decoded.split("<extra_id_0>")[1].split("<extra_id_1>")[0].strip()
else:
    span = decoded.strip()

print("Predicted fill-in for <extra_id_0>:", span)

#Load tokenized dataset (same format as training)
dataset = load_from_disk("data/tokenized_abstracts")
eval_dataset = dataset["train"].shuffle(seed=42).select(range(200, 250))  # small subset for evaluation

# Load tokenizer and fine-tuned model
model_path = "./outputs/mt5-finetuned-ftrace/checkpoint-2925855"
tokenizer = MT5Tokenizer.from_pretrained(model_path, use_fast=False)
model = MT5ForConditionalGeneration.from_pretrained(model_path)

# Prepare data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Load evaluation metrics
rouge = evaluate.load("rouge")
exact_match_metric = evaluate.load("exact_match")

# Define compute_metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    em_result = exact_match_metric.compute(predictions=decoded_preds, references=decoded_labels)
    result["exact_match"] = round(em_result["exact_match"], 4)

    print(f"Exact Match: {result['exact_match']}")
    return {k: round(v, 4) for k, v in result.items()}

# Dummy training args just for evaluation
training_args = Seq2SeqTrainingArguments(
    output_dir="./outputs/tmp-eval",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    report_to="none"
)

# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Evaluate
metrics = trainer.evaluate()
print("Evaluation Metrics:", metrics)


# Load tokenizer
tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small", use_fast=False)

# Load dataset (e.g., 50 samples from the same source)
dataset = load_from_disk("data/tokenized_abstracts")
test_dataset = dataset["train"].shuffle(seed=42).select(range(200, 250))  # same range for consistency

# Load base and fine-tuned models
base_model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
finetuned_model = MT5ForConditionalGeneration.from_pretrained("./outputs/mt5-finetuned-ftrace/checkpoint-2925855")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model.to(device).eval()
finetuned_model.to(device).eval()

# Prepare input texts and reference labels
input_texts = [tokenizer.decode(example["input_ids"], skip_special_tokens=True) for example in test_dataset]
references = [tokenizer.decode(example["labels"], skip_special_tokens=True) for example in test_dataset]

def compute_exact_match(model, input_texts, references):
    predictions = []
    for text in input_texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=128)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(decoded)
    
    result = exact_match_metric.compute(predictions=predictions, references=references)
    return round(result["exact_match"], 4)

# Compute EM for base and fine-tuned models
em_base = compute_exact_match(base_model, input_texts, references)
em_finetuned = compute_exact_match(finetuned_model, input_texts, references)

print(f"Exact Match Score BEFORE training: {em_base}")
print(f"Exact Match Score AFTER fine-tuning: {em_finetuned}")