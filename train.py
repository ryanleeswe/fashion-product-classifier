"""Train and benchmark the project's CNN and pretrained model options."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models import MODEL_NAMES, count_parameters, create_model, model_description
from train_val import train_one_epoch, validate


PROJECT_DIR = Path(__file__).resolve().parent
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FashionDataset(Dataset):
    def __init__(self, dataframe, image_dir, class_to_idx, transform):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_path = self.image_dir / f"{row['id']}.jpg"
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        image = self.transform(image)
        label = self.class_to_idx[row["articleType"]]
        return image, label


def parse_args():
    parser = argparse.ArgumentParser(
        description="Huấn luyện và so sánh các mô hình phân loại thời trang."
    )
    parser.add_argument(
        "--model",
        default="baseline",
        choices=[*MODEL_NAMES, "all"],
        help="Chọn một model hoặc 'all' để chạy benchmark mọi model.",
    )
    parser.add_argument("--csv", type=Path, default=PROJECT_DIR / "clean_dataset.csv")
    parser.add_argument("--image-dir", type=Path, default=PROJECT_DIR / "training_images")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Không tải trọng số ImageNet cho ResNet18/EfficientNet-B0.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(csv_path, image_dir, batch_size, val_size, test_size, seed, device):
    if not csv_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy CSV: {csv_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {image_dir}")

    dataframe = pd.read_csv(csv_path).dropna(subset=["id", "articleType"])
    dataframe["articleType"] = dataframe["articleType"].astype(str)
    dataframe = dataframe.drop_duplicates(subset="id").reset_index(drop=True)
    classes = sorted(dataframe["articleType"].unique().tolist())
    if len(classes) < 2:
        raise ValueError("CSV cần có ít nhất hai lớp articleType để huấn luyện.")

    counts = dataframe["articleType"].value_counts()
    too_small = counts[counts < 3]
    if not too_small.empty:
        raise ValueError(
            "Mỗi lớp cần ít nhất 3 ảnh để chia train/validation/test. "
            f"Lớp thiếu mẫu: {too_small.to_dict()}"
        )

    missing = [
        image_dir / f"{image_id}.jpg"
        for image_id in dataframe["id"]
        if not (image_dir / f"{image_id}.jpg").is_file()
    ]
    if missing:
        examples = ", ".join(path.name for path in missing[:5])
        raise FileNotFoundError(
            f"Thiếu {len(missing)} ảnh tương ứng với CSV; ví dụ: {examples}"
        )

    train_val_frame, test_frame = train_test_split(
        dataframe,
        test_size=test_size,
        random_state=seed,
        stratify=dataframe["articleType"],
    )
    train_frame, val_frame = train_test_split(
        train_val_frame,
        test_size=val_size / (1.0 - test_size),
        random_state=seed,
        stratify=train_val_frame["articleType"],
    )
    class_to_idx = {class_name: index for index, class_name in enumerate(classes)}

    # Dùng cùng kích thước và chuẩn hóa cho mọi model để so sánh trên cùng input.
    train_transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_dataset = FashionDataset(train_frame, image_dir, class_to_idx, train_transform)
    val_dataset = FashionDataset(val_frame, image_dir, class_to_idx, val_transform)
    test_dataset = FashionDataset(test_frame, image_dir, class_to_idx, val_transform)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    return train_loader, val_loader, test_loader, classes, class_to_idx


def create_optimizer(model, model_name, learning_rate, weight_decay):
    # ResNet được fine-tune với learning rate nhỏ hơn ở backbone và lớn hơn ở head.
    if model_name in {"resnet18", "resnet18_se"}:
        backbone = [
            parameter
            for name, parameter in model.named_parameters()
            if not name.startswith(("fc.", "classifier.", "channel_attention."))
        ]
        if model_name == "resnet18":
            head = list(model.fc.parameters())
        else:
            head = list(model.classifier.parameters()) + list(model.channel_attention.parameters())
        return torch.optim.AdamW(
            [
                {"params": backbone, "lr": learning_rate * 0.1},
                {"params": head, "lr": learning_rate},
            ],
            weight_decay=weight_decay,
        )
    if model_name == "efficientnet_b0":
        return torch.optim.AdamW(
            [
                {"params": model.features.parameters(), "lr": learning_rate * 0.1},
                {"params": model.classifier.parameters(), "lr": learning_rate},
            ],
            weight_decay=weight_decay,
        )

    return torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_model(
    model_name,
    args,
    train_loader,
    val_loader,
    test_loader,
    classes,
    class_to_idx,
    device,
    epoch_records,
    test_records,
):
    set_seed(args.seed)
    train_loader.generator.manual_seed(args.seed)
    model = create_model(
        model_name,
        num_classes=len(classes),
        pretrained=not args.no_pretrained,
    ).to(device)
    total_params, trainable_params = count_parameters(model)
    optimizer = create_optimizer(model, model_name, args.lr, args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    checkpoint_path = args.output_dir / f"{model_name}_best.pth"
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    seconds_to_best = 0.0
    epoch_seconds = []
    start_time = time.perf_counter()

    print(
        f"\n[{model_name}] {model_description(model_name)} | "
        f"parameters={total_params:,} trainable={trainable_params:,} | "
        f"device={device}"
    )

    for epoch in range(1, args.epochs + 1):
        synchronize(device)
        epoch_start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        synchronize(device)
        duration = time.perf_counter() - epoch_start
        epoch_seconds.append(duration)
        elapsed_seconds = time.perf_counter() - start_time
        epoch_records.append(
            {
                "model": model_name,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "epoch_seconds": round(duration, 2),
                "elapsed_seconds": round(elapsed_seconds, 2),
            }
        )

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            seconds_to_best = elapsed_seconds
            # Raw state_dict keeps the checkpoint compatible with the existing best.py.
            torch.save(model.state_dict(), checkpoint_path)
            metadata_path = args.output_dir / f"{model_name}_best.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "model_name": model_name,
                        "classes": classes,
                        "class_to_idx": class_to_idx,
                        "best_epoch": epoch,
                        "best_val_accuracy": best_val_acc,
                        "best_val_loss": best_val_loss,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        best_tag = " | saved best" if is_best else ""
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f} | "
            f"{duration:.1f}s{best_tag}"
        )

    total_seconds = time.perf_counter() - start_time
    result = {
        "model": model_name,
        "architecture": model_description(model_name),
        "pretrained": model_name in {"resnet18", "resnet18_se", "efficientnet_b0"}
        and not args.no_pretrained,
        "num_classes": len(classes),
        "parameters": total_params,
        "trainable_parameters": trainable_params,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "best_val_loss": best_val_loss,
        "seconds_to_best": round(seconds_to_best, 2),
        "average_epoch_seconds": round(sum(epoch_seconds) / len(epoch_seconds), 2),
        "total_training_seconds": round(total_seconds, 2),
        "checkpoint": str(checkpoint_path),
    }
    # Dùng test split đúng một lần sau khi đã chọn checkpoint bằng validation.
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    result["test_accuracy"] = test_acc
    result["test_loss"] = test_loss
    model.eval()
    true_labels, predicted_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            predicted = model(images.to(device)).argmax(dim=1).cpu()
            true_labels.extend(labels.tolist())
            predicted_labels.extend(predicted.tolist())
    details = classification_report(
        true_labels,
        predicted_labels,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    for class_name in classes:
        test_records.append(
            {
                "model": model_name,
                "record_type": "class_metrics",
                "true_class": class_name,
                "predicted_class": "",
                "count": details[class_name]["support"],
                "precision": details[class_name]["precision"],
                "recall": details[class_name]["recall"],
                "f1_score": details[class_name]["f1-score"],
            }
        )
    matrix = confusion_matrix(
        true_labels, predicted_labels, labels=list(range(len(classes)))
    )
    for true_index, row in enumerate(matrix):
        for predicted_index, count in enumerate(row):
            test_records.append(
                {
                    "model": model_name,
                    "record_type": "confusion_matrix",
                    "true_class": classes[true_index],
                    "predicted_class": classes[predicted_index],
                    "count": int(count),
                    "precision": "",
                    "recall": "",
                    "f1_score": "",
                }
            )
    print(
        f"[{model_name}] best epoch={best_epoch}, val_acc={best_val_acc:.4f}, "
        f"test_acc={test_acc:.4f}, time-to-best={seconds_to_best:.1f}s, "
        f"checkpoint={checkpoint_path}"
    )
    return result


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs và batch-size phải lớn hơn 0.")
    if not 0 < args.val_size < 1 or not 0 < args.test_size < 1:
        raise ValueError("val-size và test-size phải nằm giữa 0 và 1.")
    if args.val_size + args.test_size >= 1:
        raise ValueError("Tổng val-size và test-size phải nhỏ hơn 1.")

    args.csv = args.csv.resolve()
    args.image_dir = args.image_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, classes, class_to_idx = build_loaders(
        args.csv,
        args.image_dir,
        args.batch_size,
        args.val_size,
        args.test_size,
        args.seed,
        device,
    )
    print(
        f"Data: {len(train_loader.dataset)} train / "
        f"{len(val_loader.dataset)} validation / {len(test_loader.dataset)} test | "
        f"{len(classes)} classes"
    )
    print("Classes:", ", ".join(classes))

    model_names = MODEL_NAMES if args.model == "all" else (args.model,)
    results = []
    epoch_records = []
    test_records = []
    for model_name in model_names:
        results.append(
            run_model(
                model_name,
                args,
                train_loader,
                val_loader,
                test_loader,
                classes,
                class_to_idx,
                device,
                epoch_records,
                test_records,
            )
        )

    results_path = args.output_dir / "results.csv"
    with results_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nBảng kết quả đã lưu tại: {results_path}")

    epoch_metrics_path = args.output_dir / "epoch_metrics.csv"
    with epoch_metrics_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model", "epoch", "train_loss", "train_accuracy", "val_loss",
                "val_accuracy", "epoch_seconds", "elapsed_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(epoch_records)
    print(f"Chỉ số từng epoch đã lưu tại: {epoch_metrics_path}")

    test_details_path = args.output_dir / "test_details.csv"
    with test_details_path.open("w", newline="", encoding="utf-8-sig") as file:
        fields = [
            "model", "record_type", "true_class", "predicted_class", "count",
            "precision", "recall", "f1_score",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(test_records)
    print(f"Chỉ số từng lớp và confusion matrix lưu tại: {test_details_path}")


if __name__ == "__main__":
    main()
