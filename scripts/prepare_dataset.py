import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from datasets import load_dataset
from transformers import MT5Tokenizer

# Load tokenizer
tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small", use_fast=False)

# Load dataset
dataset = load_dataset("ekinakyurek/ftrace", "abstracts")

# Tokenization function
def preprocess(example):
    model_inputs = tokenizer(
        example["inputs_pretokenized"],
        max_length=128,
        padding="max_length",
        truncation=True,
    )

    labels = tokenizer(
        text_target=example["targets_pretokenized"],
        max_length=64,
        padding="max_length",
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocessing and remove old columns
tokenized_dataset = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# Save the processed dataset
tokenized_dataset.save_to_disk("data/tokenized_abstracts_128")

# Print sample
print("Tokenization complete! Sample batch:")
print(tokenized_dataset["train"][0])
