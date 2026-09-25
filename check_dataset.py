import pandas as pd
import os

base = r"C:\Users\Admin\Desktop\edho\MultiOFF_Dataset"

val = pd.read_csv(
    os.path.join(base, "Split Dataset", "Validation_meme_dataset.csv")
)

test = pd.read_csv(
    os.path.join(base, "Split Dataset", "Testing_meme_dataset.csv")
)

df = pd.concat([val, test], ignore_index=True)

missing = []

for img in df["image_name"]:
    path = os.path.join(base, "Labelled Images", img)

    if not os.path.exists(path):
        missing.append(img)

print("Total rows:", len(df))
print("Missing images:", len(missing))

if missing:
    print(missing[:20])