
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model with 3 output classes
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3
)

print("Tokenizer loaded successfully!")
print("Model loaded successfully!")
print("Number of output classes:", model.config.num_labels)
