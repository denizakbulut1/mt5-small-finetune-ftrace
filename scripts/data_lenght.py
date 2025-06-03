from datasets import load_dataset
from transformers import MT5Tokenizer
import numpy as np

dataset = load_dataset("ekinakyurek/ftrace", "abstracts")
tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small", use_fast=False)

lengths = [len(tokenizer(x)["input_ids"]) for x in dataset["train"]["inputs_pretokenized"]]

print(f"Mean: {np.mean(lengths):.2f} | Median: {np.median(lengths)} | 95th percentile: {np.percentile(lengths, 95)}")
