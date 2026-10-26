
from datasets import load_dataset
import pandas as pd

print("Loading dataset...")

dataset = load_dataset("juliensimon/amazon-shoe-reviews")

# Convert rating labels into sentiment categories
def convert_sentiment(label):
    if label in [0, 1]:
        return "negative"
    elif label == 2:
        return "neutral"
    else:
        return "positive"

# Prepare training data
train_df = pd.DataFrame(dataset["train"])
train_df = train_df.rename(
    columns={"text": "review", "labels": "rating"}
)
train_df["sentiment"] = train_df["rating"].apply(convert_sentiment)
train_df = train_df[["review", "sentiment"]]

# Prepare testing data
test_df = pd.DataFrame(dataset["test"])
test_df = test_df.rename(
    columns={"text": "review", "labels": "rating"}
)
test_df["sentiment"] = test_df["rating"].apply(convert_sentiment)
test_df = test_df[["review", "sentiment"]]

# Save the datasets
train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

print("\nDataset preparation completed!")
print("Training reviews:", len(train_df))
print("Testing reviews:", len(test_df))
print("\nTraining sentiment counts:")
print(train_df["sentiment"].value_counts())
