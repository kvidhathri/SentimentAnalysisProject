
import pandas as pd
import torch

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load a sample of the existing datasets
train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

train_df = train_df.sample(n=min(5000, len(train_df)), random_state=42)
test_df = test_df.sample(n=min(1000, len(test_df)), random_state=42)

# Prepare review text and sentiment labels
label_map = {"negative": 0, "neutral": 1, "positive": 2}

train_df = train_df.dropna(subset=["sentiment"])
test_df = test_df.dropna(subset=["sentiment"])

X_train = train_df["review"].fillna("").astype(str)
y_train = train_df["sentiment"].map(label_map)

X_test = test_df["review"].fillna("").astype(str)
y_test = test_df["sentiment"].map(label_map)

# Remove rows with labels that cannot be mapped
valid_train = y_train.notna()
valid_test = y_test.notna()

X_train, y_train = X_train[valid_train], y_train[valid_train].astype(int)
X_test, y_test = X_test[valid_test], y_test[valid_test].astype(int)

# Model 1: TF-IDF + Logistic Regression
print("Training TF-IDF + Logistic Regression...")

traditional_model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000)),
])

traditional_model.fit(X_train, y_train)
traditional_predictions = traditional_model.predict(X_test)

traditional_accuracy = accuracy_score(y_test, traditional_predictions)

print("\nTF-IDF + Logistic Regression Results")
print("Accuracy:", round(traditional_accuracy, 4))
print(classification_report(
    y_test,
    traditional_predictions,
    labels=[0, 1, 2],
    target_names=["Negative", "Neutral", "Positive"],
    zero_division=0,
))

# Model 2: Existing trained DistilBERT
print("\nEvaluating trained DistilBERT model...")

tokenizer = AutoTokenizer.from_pretrained("./sentiment_model")
model = AutoModelForSequenceClassification.from_pretrained("./sentiment_model")
model.eval()

bert_predictions = []

for start in range(0, len(X_test), 16):
    reviews = X_test.iloc[start:start + 16].tolist()

    inputs = tokenizer(
        reviews,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    batch_predictions = torch.argmax(outputs.logits, dim=1).tolist()
    bert_predictions.extend(batch_predictions)

bert_accuracy = accuracy_score(y_test, bert_predictions)

print("\nDistilBERT Results")
print("Accuracy:", round(bert_accuracy, 4))
print(classification_report(
    y_test,
    bert_predictions,
    labels=[0, 1, 2],
    target_names=["Negative", "Neutral", "Positive"],
    zero_division=0,
))

# Compare the models
print("\n========== MODEL COMPARISON ==========")
print(f"TF-IDF + Logistic Regression Accuracy: {traditional_accuracy:.4f}")
print(f"DistilBERT Accuracy:                   {bert_accuracy:.4f}")

if bert_accuracy > traditional_accuracy:
    print("Higher accuracy on this test sample: DistilBERT")
elif traditional_accuracy > bert_accuracy:
    print("Higher accuracy on this test sample: TF-IDF + Logistic Regression")
else:
    print("Both models achieved the same accuracy on this test sample.")

# Save the comparison summary
comparison_df = pd.DataFrame({
    "Model": ["TF-IDF + Logistic Regression", "DistilBERT"],
    "Accuracy": [traditional_accuracy, bert_accuracy],
})

comparison_df.to_csv("model_comparison.csv", index=False)
print("\nComparison saved to model_comparison.csv")
