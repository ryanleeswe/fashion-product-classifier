"""
train_basic.py - Train BaselineCNN 5 epoch rồi lưu baseline_cnn.pth.

Chạy:  python3 train_basic.py
Chia dữ liệu 80/10/10 GIỐNG HỆT evaluate_basic.py (random_state=42), nên tập Test
mà evaluate_basic.py dùng là ảnh mô hình chưa từng học.
Cần các file cùng thư mục: baseline_cnn.py, train_val.py, best.py
"""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from baseline_cnn import BaselineCNN
from best import fit_best

# ----------------------------------------------------------------------------
# CẤU HÌNH
# ----------------------------------------------------------------------------
LABEL_COL = "articleType"
NUM_EPOCHS = 5
SAVE_PATH = "baseline_cnn.pth"


def find_path(name):
    """Tìm file/thư mục ở thư mục hiện tại hoặc trong resources/."""
    for folder in [Path("."), Path("resources")]:
        if (folder / name).exists():
            return folder / name
    raise FileNotFoundError(f"Không tìm thấy {name} (đã tìm ở . và resources/)")


CSV_PATH = find_path("clean_dataset.csv")
IMG_DIR = find_path("training_images")

# ----------------------------------------------------------------------------
# Dữ liệu
# ----------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH).dropna(subset=["id", LABEL_COL]).reset_index(drop=True)
classes = sorted(df[LABEL_COL].unique())
class_to_idx = {name: i for i, name in enumerate(classes)}
print("Các lớp:", classes)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


class FashionDataset(Dataset):
    def __len__(self):
        return len(df)

    def __getitem__(self, i):
        row = df.iloc[i]
        with Image.open(IMG_DIR / f"{row['id']}.jpg") as img:
            image = img.convert("RGB")
        return transform(image), class_to_idx[row[LABEL_COL]]


# Chia 80/10/10, giống evaluate_basic.py
indices = list(range(len(df)))
labels = df[LABEL_COL]
train_idx, temp_idx = train_test_split(
    indices, test_size=0.2, stratify=labels, random_state=42)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5, stratify=labels.iloc[temp_idx], random_state=42)
print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)} (Test không dùng ở đây)")

dataset = FashionDataset()
train_loader = DataLoader(Subset(dataset, train_idx), batch_size=32, shuffle=True)
val_loader = DataLoader(Subset(dataset, val_idx), batch_size=32, shuffle=False)

# ----------------------------------------------------------------------------
# Train: mỗi epoch chấm Val, chỉ lưu trọng số của epoch có Val acc cao nhất
# ----------------------------------------------------------------------------
model = BaselineCNN(num_classes=len(classes))
history, best_val_acc = fit_best(
    model, train_loader, val_loader,
    num_epochs=NUM_EPOCHS, lr=1e-3, device="cpu", save_path=SAVE_PATH)

print(f"\nXong. Đã lưu mô hình tốt nhất vào: {SAVE_PATH}")
print("Giờ chạy:  python3 evaluate_basic.py")
