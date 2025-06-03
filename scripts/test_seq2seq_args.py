from transformers import Seq2SeqTrainingArguments

args = Seq2SeqTrainingArguments(
    output_dir="./test-output",
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    logging_dir="./logs"
)

print("Seq2SeqTrainingArguments accepted all parameters.")
