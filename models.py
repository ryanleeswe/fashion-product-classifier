"""Kho tạo mô hình dùng trong các thí nghiệm phân loại thời trang."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from baseline_cnn import BaselineCNN, ImprovedCNN


MODEL_NAMES = ("baseline", "improved", "resnet18", "resnet18_se", "efficientnet_b0")


class ResNet18SE(nn.Module):
    """ResNet-18 tiền huấn luyện với cơ chế chú ý kênh ở đặc trưng cuối."""

    def __init__(self, num_classes: int, pretrained: bool, dropout_rate: float) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        channels = backbone.fc.in_features
        # Chú ý kênh điều chỉnh mức quan trọng của từng đặc trưng cuối.
        hidden = max(channels // 16, 8)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate), nn.Linear(channels, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x * self.channel_attention(x).unsqueeze(-1).unsqueeze(-1)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def create_model(
    name: str,
    num_classes: int,
    pretrained: bool = True,
    dropout_rate: float = 0.3,
) -> nn.Module:
    """Tạo mô hình được yêu cầu và thay lớp phân loại theo số lớp dữ liệu."""
    name = name.lower().replace("-", "_")

    if name == "baseline":
        return BaselineCNN(num_classes=num_classes)
    if name == "improved":
        return ImprovedCNN(num_classes=num_classes, dropout_rate=dropout_rate)

    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        input_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(input_features, num_classes),
        )
        return model

    if name == "resnet18_se":
        return ResNet18SE(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate,
        )

    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        input_features = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(input_features, num_classes),
        )
        return model

    raise ValueError(
        f"Model không hợp lệ: {name}. Chọn một trong: {', '.join(MODEL_NAMES)}"
    )


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Trả về tổng số tham số và số tham số có thể huấn luyện."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def model_description(name: str) -> str:
    descriptions = {
        "baseline": "2 khối Conv + làm phẳng + Linear",
        "improved": "3 khối Conv + BatchNorm + LeakyReLU + GAP + Dropout",
        "resnet18": "ResNet-18 tiền huấn luyện trên ImageNet + đầu phân loại Dropout",
        "resnet18_se": "ResNet-18 tiền huấn luyện trên ImageNet + chú ý kênh + đầu phân loại Dropout",
        "efficientnet_b0": "EfficientNet-B0 tiền huấn luyện trên ImageNet + đầu phân loại Dropout",
    }
    return descriptions[name]
