import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

from PIL import Image

CSV_PATH = "clean_dataset_8000.csv"
IMAGE_DIR = Path("anh_tong")
OUTPUT_DIR = Path("anh_resize")
OUTPUT_CSV = "resized_dataset.csv"

IMAGE_SIZE = (128, 128)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

results = []
failed = []

##### RESIZE 

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

# Luu CSV
resized_df = pd.DataFrame(results)
resized_df.to_csv(OUTPUT_CSV, index=False)


#### CHIA TRAIN - VAL - TEST


#Chia Train 80%, con lai 20%
train_df, temp_df = train_test_split(
    resized_df,
    test_size=0.2,
    stratify=resized_df["articleType"],
    random_state=42
)

#Chia Val 10%, Test 10%
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["articleType"],
    random_state=42
)

# 4. Tao thu muc
output_dir = Path("split_dataset")
output_dir.mkdir(exist_ok=True)

# 5. Luu CSV
train_df.to_csv(output_dir / "train.csv", index=False)
val_df.to_csv(output_dir / "val.csv", index=False)
test_df.to_csv(output_dir / "test.csv", index=False)


####### ARGUMENTATION CHO TRAIN_ 

from torchvision import transforms
train_transform = transforms.Compose([
    # Xoay ngau nhien
    transforms.RandomRotation(
        degrees=20
    ),

    #Lat ngang voi xac suat 50%
    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    # Thay doi mau sac va anh sang
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),
   

    # che khuat ngau nhien
    transforms.RandomErasing(
        p=0.5,
        scale=(0.02, 0.15),
        ratio=(0.3, 3.3),
        value=0
    ),
 #  Chuyen anh thanh Tensor
    transforms.ToTensor(),
    # Normalize
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# VALIDATION 
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# TEST
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])
