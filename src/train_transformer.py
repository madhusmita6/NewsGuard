import os
import pandas as pd
import torch
import transformers
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
import mlflow
import mlflow.transformers
# === Setup paths ===
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_df = pd.read_csv(os.path.join(base_dir, "data", "processed", "train.csv"))
test_df = pd.read_csv(os.path.join(base_dir, "data", "processed", "test.csv"))

# === Clean and prepare ===
train_df = train_df[["text", "label"]].dropna()
test_df = test_df[["text", "label"]].dropna()
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# === Tokenizer and model ===
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# === Metrics ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions)
    }

# === Training args ===
output_dir = os.path.join(base_dir, "models", "roberta")
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir=os.path.join(base_dir, "logs"),
    logging_steps=50,
)

# === MLflow logging ===
mlflow.set_experiment("NewsGuard - RoBERTa")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model", "roberta-base")
    mlflow.log_param("epochs", training_args.num_train_epochs)
    mlflow.log_param("batch_size", training_args.per_device_train_batch_size)
    mlflow.log_param("max_length", 512)

    # Train with Trainer API
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Evaluate on test set
    eval_results = trainer.evaluate()
    mlflow.log_metrics({
        "accuracy": eval_results["eval_accuracy"],
        "f1": eval_results["eval_f1"]
    })

    # Save model artifacts
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Log model to MLflow
    mlflow.transformers.log_model(
        transformers_model=model,
        artifact_path="model",
        tokenizer=tokenizer,
        input_example={"text": "Sample news article text here."}
    )

print("✅ RoBERTa model trained, saved, and logged to MLflow.")
