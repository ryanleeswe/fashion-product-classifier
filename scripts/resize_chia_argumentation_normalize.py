"""Thí nghiệm resize ảnh, chia train/validation/test và định nghĩa augmentation."""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_DIR / "clean_dataset.csv"
IMAGE_DIR = PROJECT_DIR / "training_images"
OUTPUT_DIR = PROJECT_DIR / "training_images_resized"
OUTPUT_CSV = PROJECT_DIR / "resized_clean_dataset.csv"

IMAGE_SIZE = (128, 128)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

results = []
failed = []

# Resize ảnh về kích thước đầu vào thống nhất.

for _, row in df.iterrows():

    image_id = int(row["id"])

    source = IMAGE_DIR / f"{image_id}.jpg"
    destination = OUTPUT_DIR / f"{image_id}.jpg"

    try:
        with Image.open(source) as image:
            image = image.convert("RGB")
            image = image.resize(
                IMAGE_SIZE,
                Image.Resampling.LANCZOS
            )
            image.save(destination, quality=95)

        results.append(row.to_dict())

    except Exception:
        failed.append(image_id)

# Lưu CSV sau khi resize thành công.
resized_df = pd.DataFrame(results)
resized_df.to_csv(OUTPUT_CSV, index=False)


# Chia dữ liệu thành train, validation và test.


# Dành 80% cho train và 20% cho tập còn lại.
train_df, temp_df = train_test_split(
    resized_df,
    test_size=0.2,
    stratify=resized_df["articleType"],
    random_state=42
)

# Chia phần còn lại thành validation 10% và test 10%.
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["articleType"],
    random_state=42
)

# Tạo thư mục lưu các CSV đã chia.
output_dir = PROJECT_DIR / "split_dataset"
output_dir.mkdir(exist_ok=True)

# 5. Luu CSV
train_df.to_csv(output_dir / "train.csv", index=False)
val_df.to_csv(output_dir / "val.csv", index=False)
test_df.to_csv(output_dir / "test.csv", index=False)


# Định nghĩa augmentation và chuẩn hóa cho train.

from torchvision import transforms
train_transform = transforms.Compose([
    # Xoay ảnh ngẫu nhiên.
    transforms.RandomRotation(
        degrees=20
    ),

    # Lật ngang với xác suất 50%.
    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    # Thay đổi màu sắc và độ sáng.
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),

    # Che ngẫu nhiên một vùng của ảnh.
    transforms.RandomErasing(
        p=0.5,
        scale=(0.02, 0.15),
        ratio=(0.3, 3.3),
        value=0
    ),
    # Chuyển ảnh thành tensor.
    transforms.ToTensor(),
    # Chuẩn hóa tensor.
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# Tập validation chỉ resize, chuyển tensor và chuẩn hóa.
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# Tập test chỉ chuyển tensor và chuẩn hóa.
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])
