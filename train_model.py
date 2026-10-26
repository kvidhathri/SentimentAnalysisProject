
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score

model_name = "distilbert-base-uncased"

# Load the larger datasets
train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

# Use smaller samples for the first CPU training run
train_df = train_df.sample(n=5000, random_state=42)
test_df = test_df.sample(n=1000, random_state=42)

label_map = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

train_df["label"] = train_df["sentiment"].map(label_map)
test_df["label"] = test_df["sentiment"].map(label_map)

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_data(batch):
    reviews = [
        str(review) if review is not None else ""
        for review in batch["review"]
    ]

    return tokenizer(
        reviews,
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_dataset = Dataset.from_pandas(
    train_df[["review", "label"]], preserve_index=False
)
test_dataset = Dataset.from_pandas(
    test_df[["review", "label"]], preserve_index=False
)

train_dataset = train_dataset.map(tokenize_data, batched=True)
test_dataset = test_dataset.map(tokenize_data, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    id2label={0: "negative", 1: "neutral", 2: "positive"},
    label2id=label_map
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(
            labels, predictions,
            average="macro",
            zero_division=0
        )
    }

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="no",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

print("\nModel evaluation:")
print(trainer.evaluate())

trainer.save_model("./sentiment_model")
tokenizer.save_pretrained("./sentiment_model")

print("\nTraining completed!")
print("Model saved in the sentiment_model folder.")
