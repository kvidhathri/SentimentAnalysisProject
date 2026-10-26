from datasets import load_dataset

dataset = load_dataset("juliensimon/amazon-shoe-reviews")

print("Training label values:")
print(sorted(set(dataset["train"]["labels"])))

print("\nTest label values:")
print(sorted(set(dataset["test"]["labels"])))

print("\nExample reviews:")
for i in range(5):
    print(
        "Label:", dataset["train"][i]["labels"],
        "| Review:", dataset["train"][i]["text"]
    )

 
