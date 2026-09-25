from src.dataset import get_datasets

train_ds, val_ds, test_ds = get_datasets()

print("Train:", len(train_ds))
print("Val:", len(val_ds))
print("Test:", len(test_ds))

sample = train_ds[0]

print(sample["image"].shape)
print(sample["input_ids"].shape)
print(sample["label"])