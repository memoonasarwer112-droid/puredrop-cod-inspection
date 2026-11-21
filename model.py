import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class BaselineCODModel(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = resnet50(weights=None)

        self.encoder = nn.Sequential(
            *list(backbone.children())[:-2]
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                2048, 512,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                512, 1,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        original_size = x.shape[2:]

        features = self.encoder(x)
        output = self.decoder(features)

        output = F.interpolate(
            output,
            size=original_size,
            mode="bilinear",
            align_corners=False
        )

        return output