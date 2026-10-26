
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "./sentiment_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

labels = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

review = input("Enter a product review: ")

inputs = tokenizer(
    review,
    return_tensors="pt",
    truncation=True,
    max_length=128
)

with torch.no_grad():
    outputs = model(**inputs)

probabilities = torch.softmax(outputs.logits, dim=1)[0]
prediction = torch.argmax(probabilities).item()

print("\nPredicted sentiment:", labels[prediction])
print("Confidence:", f"{probabilities[prediction].item() * 100:.2f}%")
