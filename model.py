import timm
import torch
from torch import nn


class MaskedViTFL(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(
            "vit_small_patch16_224", pretrained=pretrained, num_classes=0
        )
        dim = self.encoder.num_features
        self.mask = nn.Parameter(torch.ones(dim))
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, images):
        features = self.encoder.forward_features(images)
        if features.ndim == 3:
            features = features[:, 0]  # CLS token
        return self.classifier(features * self.mask)
