import torch
import torch.nn as nn

from train_val import train_one_epoch, validate


def fit_best(model, train_loader, val_loader, num_epochs=15, lr=1e-3,
             device=None, save_path="best_model.pth"):
    """Train nhiều epoch, chỉ lưu lại trọng số của epoch có val accuracy cao nhất.

    Trả về history (để vẽ biểu đồ) và best_val_acc.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_epoch = 0.0, 0

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = validate(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        msg = (f"Epoch {epoch}/{num_epochs} | "
               f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
               f"val loss {va_loss:.4f} acc {va_acc:.4f}")

        # Bắt khoảnh khắc tốt nhất: val acc cao hơn mọi epoch trước thì lưu lại
        if va_acc > best_val_acc:
            best_val_acc, best_epoch = va_acc, epoch
            torch.save(model.state_dict(), save_path)
            msg += "  <-- best, đã lưu"

        print(msg)

    print(f"\nTốt nhất: epoch {best_epoch}, val acc {best_val_acc:.4f} -> {save_path}")
    return history, best_val_acc


@torch.no_grad()
def collect_predictions(model, loader, device=None):
    """Thu thập nhãn thật và nhãn dự đoán trên cả tập loader.

    Dùng để tính confusion matrix, precision/recall (sklearn) ở giai đoạn đánh giá.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    for images, labels in loader:
        outputs = model(images.to(device))
        y_pred.extend(outputs.argmax(dim=1).cpu().tolist())
        y_true.extend(labels.tolist())

    return y_true, y_pred


def load_best(model, save_path="best_model.pth", device=None):
    """Nạp lại trọng số tốt nhất vào model."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model.to(device)
