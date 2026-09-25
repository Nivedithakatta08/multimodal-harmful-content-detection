import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from transformers import DistilBertModel


class MultimodalModel(nn.Module):

    def __init__(self, num_classes=2):

        super().__init__()

        self.image_model = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        for param in self.image_model.parameters():
            param.requires_grad = False

        image_features = self.image_model.fc.in_features

        self.image_model.fc = nn.Identity()

        self.text_model = DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )

        for param in self.text_model.parameters():
            param.requires_grad = False

        text_features = self.text_model.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(image_features + text_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(
        self,
        image,
        input_ids,
        attention_mask
    ):

        with torch.no_grad():
            image_features = self.image_model(image)

            text_output = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        text_features = text_output.last_hidden_state[:, 0, :]

        combined = torch.cat(
            [image_features, text_features],
            dim=1
        )

        output = self.classifier(combined)

        return output