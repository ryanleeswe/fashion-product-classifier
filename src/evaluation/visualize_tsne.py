"""Tạo biểu đồ t-SNE hai chiều từ đặc trưng lớp trước lớp phân loại."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import transforms

from src.evaluation.evaluate import IMG_SIZE, ImageDataset, build_model, split_train_val_test


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="baseline", choices=["baseline", "improved", "resnet18"])
    parser.add_argument("--weights", default="baseline_cnn.pth")
    parser.add_argument("--csv", default="clean_dataset.csv")
    parser.add_argument("--img_dir", default="training_images")
    parser.add_argument("--test_csv", default=None, help="CSV test đã tách sẵn; nếu bỏ trống sẽ dùng seed=42 như evaluate.py.")
    parser.add_argument("--out", default="eval_output/tsne.png")
    parser.add_argument("--max_per_class", type=int, default=300, help="Giới hạn mẫu/lớp để t-SNE chạy nhanh hơn; 0 dùng toàn bộ.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imagenet_norm", action="store_true")
    args = parser.parse_args()

    # Đọc dữ liệu và thống nhất thứ tự chỉ số lớp trong toàn bộ quá trình chạy.
    df = pd.read_csv(args.csv).dropna(subset=["id", "articleType"]).reset_index(drop=True)
    classes = sorted(df["articleType"].astype(str).unique())
    class_to_idx = {name: index for index, name in enumerate(classes)}

    # Dùng test CSV có sẵn nếu được truyền vào; nếu không thì tạo lại cùng split seed=42.
    if args.test_csv:
        test_df = pd.read_csv(args.test_csv).dropna(subset=["id", "articleType"])
    else:
        _, _, test_indices = split_train_val_test(df, "articleType", seed=args.seed)
        test_df = df.iloc[test_indices]
    test_df = test_df.copy()
    test_df["articleType"] = test_df["articleType"].astype(str)

    # Giới hạn số ảnh mỗi lớp để t-SNE chạy nhanh và tránh tập dữ liệu không cân bằng chi phối biểu đồ.
    if args.max_per_class > 0:
        test_df = pd.concat(
            [
                part.sample(min(len(part), args.max_per_class), random_state=args.seed)
                for _, part in test_df.groupby("articleType", sort=False)
            ],
            ignore_index=True,
        )
    test_df = test_df.reset_index(drop=True)

    transforms_list = [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]
    if args.imagenet_norm:
        transforms_list.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    dataset = ImageDataset(test_df, args.img_dir, class_to_idx, "articleType", transforms.Compose(transforms_list))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, len(classes))
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device).eval()

    # Chọn lớp classification head tương ứng để đăng ký hook lấy đặc trưng trước khi phân loại.
    if args.model in ("baseline", "improved"):
        target = model.classifier
    elif args.model == "resnet18":
        target = model.fc
    else:
        target = model.fc

    # Hook lưu vector đầu vào của lớp classification head cho từng batch ảnh.
    feature_batches: list[torch.Tensor] = []
    def capture_features(_module, inputs):
        feature_batches.append(inputs[0].detach().flatten(start_dim=1).cpu())
    hook = target.register_forward_pre_hook(capture_features)
    try:
        with torch.inference_mode():
            for images, _labels, _indices in loader:
                model(images.to(device))
    finally:
        hook.remove()

    features = torch.cat(feature_batches).numpy()
    labels = test_df["articleType"].to_numpy()
    if len(features) < 3:
        raise ValueError("Cần ít nhất 3 ảnh để tạo biểu đồ t-SNE.")

    # Kích thước tối đa hợp lệ của perplexity và PCA được lấy từ dữ liệu hiện tại.
    perplexity = min(30.0, float(len(features) - 1))
    pca_dimensions = min(50, features.shape[0], features.shape[1])
    reduced_features = PCA(n_components=pca_dimensions, svd_solver="randomized",
                            random_state=args.seed).fit_transform(features)
    points = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=args.seed).fit_transform(reduced_features)

    # Vẽ từng lớp bằng một màu riêng và hiển thị số lượng ảnh tương ứng.
    fig, ax = plt.subplots(figsize=(11, 8))
    palette = plt.get_cmap("tab10")
    for index, class_name in enumerate(classes):
        mask = labels == class_name
        if mask.any():
            ax.scatter(points[mask, 0], points[mask, 1], s=18, alpha=0.75,
                       color=palette(index % 10), label=f"{class_name} (n={mask.sum()})")
    ax.set_title(f"Đặc trưng lớp trước phân loại — {args.model} ({len(features)} ảnh test)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best", fontsize="small", frameon=True)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Đã lưu {output} ({len(features)} ảnh; perplexity={perplexity:g})")


if __name__ == "__main__":
    main()
