import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

# 1. KHAI BÁO
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "clean_dataset.csv")
IMG_DIR = os.path.join(BASE_DIR, "training_images")

df = pd.read_csv(CSV_PATH)

# 2. ĐỊNH NGHĨA TRANSFORMS

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

#3. XÂY DỰNG DATASET

class FashionDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

        # Tạo danh sách tên các lớp và ánh xạ sang số
        self.classes = sorted(self.df["articleType"].unique())
        self.class_to_idx = {
            class_name: index
            for index, class_name in enumerate(self.classes)
        }
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Tìm file ảnh theo ID trong CSV
        img_filename = f"{row['id']}.jpg"
        img_path = os.path.join(self.img_dir, img_filename)
        # Đọc ảnh và chuyển sang RGB
        with Image.open(img_path) as img:
            image = img.convert("RGB")
        # Resize và chuyển ảnh thành Tensor
        if self.transform is not None:
            image = self.transform(image)
        # Chuyển nhãn dạng chữ thành số
        label = self.class_to_idx[row["articleType"]]
        return image, label

#4. KHỞI TẠO DATASET
dataset = FashionDataset(
    dataframe=df,
    img_dir=IMG_DIR,
    transform=transform
)

#5. CHIA TRAIN/VAL: 80 / 20
indices = list(range(len(dataset)))
labels = df["articleType"].map(
    dataset.class_to_idx
).tolist()

train_indices, val_indices = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
#6. TẠO DATALOADER
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)
#7. HIỂN THỊ RIÊNG TỪNG ẢNH ĐỂ KIỂM TRA
# Lấy một batch từ tập Train
images, batch_labels = next(iter(train_loader))
num_images = min(5, len(images))

for i in range(num_images):
    # Tensor ban đầu có dạng (C, H, W)
    image_tensor = images[i]
    # Chuyển thành dạng (H, W, C) để matplotlib hiển thị
    image_show = image_tensor.permute(1, 2, 0).numpy()
    # Hiển thị từng ảnh trong một cửa sổ riêng
    plt.figure(figsize=(5, 5))
    plt.imshow(image_show)
    plt.title(
        f"{dataset.classes[batch_labels[i].item()]}\n"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show()
