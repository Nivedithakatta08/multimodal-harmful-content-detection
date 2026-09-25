import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from transformers import DistilBertTokenizer
from torchvision import transforms


DATASET_ROOT = r"MultiOFF_Dataset"

IMAGE_DIR = os.path.join(
    DATASET_ROOT,
    "Labelled Images"
)

VAL_CSV = os.path.join(
    DATASET_ROOT,
    "Split Dataset",
    "Validation_meme_dataset.csv"
)

TEST_CSV = os.path.join(
    DATASET_ROOT,
    "Split Dataset",
    "Testing_meme_dataset.csv"
)


class MultiOFFDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer,
        transform
    ):

        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = os.path.join(
            IMAGE_DIR,
            row["image_name"]
        )

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        text = str(row["sentence"])

        encoding = self.tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        label = 1 if row["label"] == "offensive" else 0

        return {
            "image": image,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(
                label,
                dtype=torch.long
            )
        }


def get_datasets():

    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert-base-uncased"
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    val_df = pd.read_csv(VAL_CSV)

    train_df, valid_df = train_test_split(
        val_df,
        test_size=0.2,
        random_state=42,
        stratify=val_df["label"]
    )

    test_df = pd.read_csv(TEST_CSV)

    train_dataset = MultiOFFDataset(
        train_df,
        tokenizer,
        transform
    )

    valid_dataset = MultiOFFDataset(
        valid_df,
        tokenizer,
        transform
    )

    test_dataset = MultiOFFDataset(
        test_df,
        tokenizer,
        transform
    )

    return (
        train_dataset,
        valid_dataset,
        test_dataset
    )