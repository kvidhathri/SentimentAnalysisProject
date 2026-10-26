
import pandas as pd
from sklearn.model_selection import train_test_split

# Load cleaned data
df = pd.read_csv("product_reviews_cleaned.csv")

# Split the data into training and testing sets
train_data, test_data = train_test_split(
    df,
    test_size=0.25,
    random_state=42
)

# Save both datasets
train_data.to_csv("train_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)

print("Data splitting completed!")
print("Training reviews:", len(train_data))
print("Testing reviews:", len(test_data))
