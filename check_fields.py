from datasets import load_dataset

print("Loading dataset...")

ds = load_dataset(
    "neuralcatcher/hateful_memes",
    split="train[:1]"
)

print("Loaded")
print(ds.features)
print(ds.column_names)
print(ds[0])