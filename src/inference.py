import sys
import torch
from PIL import Image
from transformers import DistilBertTokenizer
from torchvision import transforms

from src.model import MultimodalModel


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "checkpoints/model_epoch_5.pth"

image_path = sys.argv[1]
text = sys.argv[2]

tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0)

encoding = tokenizer(
    text,
    max_length=128,
    padding="max_length",
    truncation=True,
    return_tensors="pt"
)

model = MultimodalModel()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model = model.to(DEVICE)
model.eval()

with torch.no_grad():

    output = model(
        image.to(DEVICE),
        encoding["input_ids"].to(DEVICE),
        encoding["attention_mask"].to(DEVICE)
    )

    prediction = torch.argmax(
        output,
        dim=1
    ).item()

if prediction == 1:
    print("Prediction: Offensive")
else:
    print("Prediction: Non-offensive")