
from transformers import AutoTokenizer

# Load the DistilBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

# Sample product review
review = "This product is excellent"

# Convert text into tokens
tokens = tokenizer(
    review,
    padding="max_length",
    truncation=True,
    max_length=128
)

print("Original review:", review)
print("Token IDs:", tokens["input_ids"])
print("Attention mask:", tokens["attention_mask"])
