
import pandas as pd
import re

# Read the raw review data
df = pd.read_csv("product_reviews_raw.csv")

# Remove missing and duplicate reviews
df = df.dropna()
df = df.drop_duplicates()

# Clean the review text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["review"] = df["review"].apply(clean_text)

# Save the cleaned data
df.to_csv("product_reviews_cleaned.csv", index=False)

print("Data preparation completed!")
print(df.head())
print("\nTotal reviews:", len(df))
