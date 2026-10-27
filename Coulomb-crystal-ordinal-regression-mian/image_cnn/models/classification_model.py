import torch
from torch import nn
from torchvision import models


class ClassificationNet(nn.Module):
    """ResNet18 with an 80-class softmax head."""

    def __init__(self, weights: str = "default") -> None:
        super().__init__()
        pretrained = models.ResNet18_Weights.DEFAULT if weights == "default" else None
        network = models.resnet18(weights=pretrained)
        original_conv = network.conv1
        network.conv1 = nn.Conv2d(
            1,
            original_conv.out_channels,
            original_conv.kernel_size,
            original_conv.stride,
            original_conv.padding,
            bias=False,
        )
        if pretrained is not None:
            with torch.no_grad():
                network.conv1.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))
        network.fc = nn.Linear(network.fc.in_features, 80)
        self.net = network

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)
