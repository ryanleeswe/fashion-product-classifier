"""Mô hình CNN cơ sở để phân loại ảnh thời trang thành bốn loại."""

import torch
from torch import nn


class BaselineCNN(nn.Module):
    """Phân loại ảnh RGB kích thước 128x128 thành các lớp sản phẩm."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()

        # Các lớp trích xuất đặc trưng từ ảnh đầu vào.
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Lớp phân loại chuyển đặc trưng thành điểm số cho từng lớp.
        self.classifier = nn.Linear(32 * 32 * 32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Trích xuất đặc trưng, làm phẳng và phân loại ảnh.
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


if __name__ == "__main__":
    # Kiểm tra nhanh mô hình với một ảnh giả.
    model = BaselineCNN()
    dummy_image = torch.randn(1, 3, 128, 128)
    output = model(dummy_image)
    print(f"Output shape: {tuple(output.shape)}")
