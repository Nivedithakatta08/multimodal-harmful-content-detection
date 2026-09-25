import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import get_datasets
from src.model import MultimodalModel


BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

os.makedirs("checkpoints", exist_ok=True)


train_dataset, val_dataset, test_dataset = get_datasets()

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


model = MultimodalModel()
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE
)


best_val_accuracy = 0.0


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for batch in train_loader:

        images = batch["image"].to(DEVICE)
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            images,
            input_ids,
            attention_mask
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    train_loss = total_loss / len(train_loader)
    train_accuracy = correct / total

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for batch in val_loader:

            images = batch["image"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(
                images,
                input_ids,
                attention_mask
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_accuracy = val_correct / val_total

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.4f} "
        f"Val Acc: {val_accuracy:.4f}"
    )

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "checkpoints/best_model.pth"
        )

print("Training complete.")