from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="test_output",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    eval_strategy="epoch"
)

print("✅ TrainingArguments works correctly.")
