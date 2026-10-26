
import pandas as pd
import torch
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Load the test dataset
df = pd.read_csv("test_data.csv")

# Evaluate on 1,000 reviews to keep the CPU run manageable
df = df.sample(n=min(1000, len(df)), random_state=42).reset_index(drop=True)

# Convert sentiment labels into numbers
label_map = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

y_true = df["sentiment"].map(label_map).tolist()

# Load the trained model
tokenizer = AutoTokenizer.from_pretrained("./sentiment_model")
model = AutoModelForSequenceClassification.from_pretrained(
    "./sentiment_model"
)
model.eval()

# Predict in small batches
predictions = []

for start in range(0, len(df), 16):
    reviews = (
        df["review"]
        .iloc[start:start + 16]
        .fillna("")
        .astype(str)
        .tolist()
    )

    inputs = tokenizer(
        reviews,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    batch_predictions = torch.argmax(
        outputs.logits, dim=1
    ).tolist()

    predictions.extend(batch_predictions)

# Print evaluation metrics
print("\nAccuracy:", round(accuracy_score(y_true, predictions), 4))

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        predictions,
        labels=[0, 1, 2],
        target_names=["Negative", "Neutral", "Positive"],
        zero_division=0,
    )
)

# Create and save the confusion matrix
cm = confusion_matrix(y_true, predictions, labels=[0, 1, 2])

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Neutral", "Positive"],
)

display.plot()
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)

print("\nConfusion matrix saved as confusion_matrix.png")

plt.show()
