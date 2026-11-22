import torch
import torch.nn as nn


def accuracy(outputs, labels):
    """Số mẫu đoán đúng trong 1 batch."""
    return (outputs.argmax(dim=1) == labels).sum().item()


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Huấn luyện 1 epoch. Trả về (loss trung bình, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()              # 1. xóa gradient cũ
        outputs = model(images)            # 2. forward
        loss = criterion(outputs, labels)  # 3. tính loss
        loss.backward()                    # 4. backward
        optimizer.step()                   # 5. cập nhật trọng số

        total_loss += loss.item() * images.size(0)
        correct += accuracy(outputs, labels)
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Đánh giá trên tập validation (không cập nhật trọng số)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct += accuracy(outputs, labels)
        total += labels.size(0)

    return total_loss / total, correct / total


def fit(model, train_loader, val_loader, num_epochs=10, lr=1e-3, device=None):
    """Chạy nhiều epoch, in kết quả mỗi epoch."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}")