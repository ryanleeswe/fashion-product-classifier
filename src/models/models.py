"""Kho tạo mô hình dùng trong phân loại thời trang."""

from __future__ import annotations

from torch import nn
from torchvision import models

from baseline_cnn import BaselineCNN, ImprovedCNN


MODEL_NAMES = ("baseline", "improved", "resnet18")


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
        "improved": "3 khối Conv + BatchNorm + LeakyReLU + AdaptiveAvgPool 4x4 + Dropout",
        "resnet18": "ResNet-18 tiền huấn luyện trên ImageNet + đầu phân loại Dropout",
    }
    return descriptions[name]
