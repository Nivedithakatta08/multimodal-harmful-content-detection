import torch
from src.model import MultimodalModel

model = MultimodalModel()

image = torch.randn(2, 3, 224, 224)
input_ids = torch.randint(0, 30522, (2, 128))
attention_mask = torch.ones(2, 128)

output = model(
    image,
    input_ids,
    attention_mask
)

print("Output shape:", output.shape)