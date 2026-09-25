import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.dataset import get_datasets
from src.model import MultimodalModel


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

_, _, test_dataset = get_datasets()

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False
)

model = MultimodalModel()

model.load_state_dict(
    torch.load(
        "checkpoints/best_model.pth",
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for batch in test_loader:

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

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print("Test Accuracy:", accuracy)

print("\nClassification Report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=[
            "Non-offensive",
            "Offensive"
        ]
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        all_labels,
        all_predictions
    )
)