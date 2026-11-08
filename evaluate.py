"""
evaluate.py - Đánh giá mô hình tốt nhất trên tập Test.

Làm 4 việc:
  1. Chạy toàn bộ tập Test qua mô hình, so nhãn dự đoán với nhãn thật.
  2. Xuất confusion matrix + classification report (precision, recall, F1).
  3. Grad-CAM: vẽ vùng nhiệt cho ảnh đoán ĐÚNG và đoán SAI.
  4. Gom tất cả ảnh đoán sai vào một thư mục riêng để nhận xét và cải tiến.

Ví dụ chạy:
  python evaluate.py --model improved --weights best_model.pth
  python evaluate.py --model baseline --weights baseline_cnn.pth
  python evaluate.py --model resnet18 --weights best_resnet.pth --imagenet_norm

Kết quả nằm trong thư mục --out (mặc định eval_output).
"""

import argparse
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # chỉ lưu file, không mở cửa sổ
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from baseline_cnn import BaselineCNN, ImprovedCNN

IMG_SIZE = 128
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ----------------------------------------------------------------------------
# 1. Dữ liệu: chia train/val/test và Dataset
# ----------------------------------------------------------------------------
def split_train_val_test(df, label_col, seed=42):
    """Chia 80/10/10, giữ nguyên tỷ lệ các lớp (stratify).

    QUAN TRỌNG: file train phải dùng đúng hàm chia này (cùng seed) thì tập Test
    mới thật sự là ảnh mô hình chưa từng thấy.
    """
    labels = df[label_col]
    idx = np.arange(len(df))
    train_idx, temp_idx = train_test_split(
        idx, test_size=0.2, stratify=labels, random_state=seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=labels.iloc[temp_idx], random_state=seed
    )
    return train_idx, val_idx, test_idx


class ImageDataset(Dataset):
    """Trả về (ảnh đã xử lý, nhãn số, vị trí dòng trong dataframe)."""

    def __init__(self, df, img_dir, class_to_idx, label_col, tfm):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.class_to_idx = class_to_idx
        self.label_col = label_col
        self.tfm = tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        with Image.open(self.img_dir / f"{row['id']}.jpg") as im:
            img = im.convert("RGB")
        return self.tfm(img), self.class_to_idx[row[self.label_col]], i


# ----------------------------------------------------------------------------
# 2. Khởi tạo mô hình và chọn lớp để làm Grad-CAM
# ----------------------------------------------------------------------------
def build_model(name, num_classes):
    if name == "baseline":
        return BaselineCNN(num_classes)
    if name == "improved":
        return ImprovedCNN(num_classes)
    if name == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    raise ValueError(f"Chưa hỗ trợ mô hình: {name}")


def get_target_layer(model, name):
    """Lớp tích chập CUỐI CÙNG (còn giữ thông tin vị trí trong ảnh)."""
    if name == "baseline":
        return model.features[4]    # ReLU sau Conv thứ 2 -> 32x32x32
    if name == "improved":
        return model.features[10]   # LeakyReLU sau Conv thứ 3 -> 128x32x32
    if name == "resnet18":
        return model.layer4[-1]
    if name == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(name)


# ----------------------------------------------------------------------------
# 3. Dự đoán toàn bộ tập Test
# ----------------------------------------------------------------------------
@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    probs, labels, idxs = [], [], []
    for x, y, i in loader:
        logits = model(x.to(device))
        probs.append(torch.softmax(logits, dim=1).cpu())
        labels.append(y)
        idxs.append(i)
    return (
        torch.cat(probs).numpy(),
        torch.cat(labels).numpy(),
        torch.cat(idxs).numpy(),
    )


# ----------------------------------------------------------------------------
# 4. Confusion matrix
# ----------------------------------------------------------------------------
def plot_confusion_matrix(cm, classes, path):
    cm_pct = cm / cm.sum(axis=1, keepdims=True).clip(min=1) * 100
    n = len(classes)
    size = max(5, 0.9 * n + 2)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Nhãn dự đoán")
    ax.set_ylabel("Nhãn thật")
    ax.set_title("Confusion matrix (số ảnh và % theo hàng)")
    for r in range(n):
        for c in range(n):
            color = "white" if cm_pct[r, c] > 50 else "black"
            ax.text(c, r, f"{cm[r, c]}\n{cm_pct[r, c]:.1f}%",
                    ha="center", va="center", color=color, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def top_confusions(cm, classes, k=5):
    pairs = []
    for r in range(len(classes)):
        for c in range(len(classes)):
            if r != c and cm[r, c] > 0:
                pairs.append((cm[r, c], classes[r], classes[c]))
    return sorted(pairs, reverse=True)[:k]


# ----------------------------------------------------------------------------
# 5. Grad-CAM
# ----------------------------------------------------------------------------
class GradCAM:
    """Grad-CAM: vùng nào của ảnh làm điểm của một lớp tăng lên nhiều nhất."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._on_forward)

    def _on_forward(self, module, inputs, output):
        self.activations = output
        output.register_hook(self._on_backward)  # bắt gradient của đầu ra lớp này

    def _on_backward(self, grad):
        self.gradients = grad

    def __call__(self, x, class_idx):
        """x: tensor 1xCxHxW. Trả về heatmap HxW trong [0, 1] cho lớp class_idx."""
        self.model.eval()
        self.model.zero_grad()
        with torch.enable_grad():
            logits = self.model(x)
            logits[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # tầm quan trọng mỗi kênh
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


def overlay_heatmap(img_np, cam, alpha=0.45):
    heat = plt.get_cmap("jet")(cam)[..., :3]
    return np.clip((1 - alpha) * img_np + alpha * heat, 0, 1)


def load_display_image(path):
    """Ảnh gốc resize 128x128, giá trị 0..1, dùng để vẽ (không normalize)."""
    with Image.open(path) as im:
        im = im.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(im, dtype=np.float32) / 255.0


def make_input(img_np, imagenet_norm, device):
    x = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    if imagenet_norm:
        x = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(x)
    return x.to(device)


def save_gradcam_case(cam_engine, row, img_dir, classes, out_dir, imagenet_norm, device):
    """Lưu overlay riêng (để dựng giao diện) và một hình tổng hợp có tiêu đề.

    Ảnh đoán sai có thêm cột Grad-CAM của LỚP THẬT: đối chiếu mô hình đã nhìn
    vào đâu với chỗ đáng lẽ phải nhìn.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    img_id = row["id"]
    img_np = load_display_image(Path(img_dir) / f"{img_id}.jpg")
    x = make_input(img_np, imagenet_norm, device)

    pred_idx, true_idx = int(row["pred_idx"]), int(row["true_idx"])
    cam_pred = cam_engine(x, pred_idx)
    ov_pred = overlay_heatmap(img_np, cam_pred)
    plt.imsave(out_dir / f"{img_id}_pred.png", ov_pred)

    panels = [(img_np, "Ảnh gốc"),
              (ov_pred, f"Vùng nhìn để đoán: {classes[pred_idx]}")]
    paths = {"overlay_pred": str(out_dir / f"{img_id}_pred.png"), "overlay_true": ""}

    if not row["correct"]:
        cam_true = cam_engine(x, true_idx)
        ov_true = overlay_heatmap(img_np, cam_true)
        plt.imsave(out_dir / f"{img_id}_true.png", ov_true)
        panels.append((ov_true, f"Vùng của lớp thật: {classes[true_idx]}"))
        paths["overlay_true"] = str(out_dir / f"{img_id}_true.png")

    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (im, title) in zip(axes, panels):
        ax.imshow(im)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    status = "ĐÚNG" if row["correct"] else "SAI"
    fig.suptitle(f"id {img_id} | thật: {classes[true_idx]} | đoán: {classes[pred_idx]} "
                 f"({row['confidence']:.1%}) | {status}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f"{img_id}_figure.png", dpi=120)
    plt.close(fig)
    return paths


# ----------------------------------------------------------------------------
# 6. Chương trình chính
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="improved",
                   choices=["baseline", "improved", "resnet18", "efficientnet_b0"])
    p.add_argument("--weights", default="best_model.pth")
    p.add_argument("--csv", default="clean_dataset.csv")
    p.add_argument("--img_dir", default="training_images")
    p.add_argument("--label_col", default="articleType")
    p.add_argument("--test_csv", default=None,
                   help="Nếu đã có sẵn file CSV của tập Test thì dùng, không tự chia nữa")
    p.add_argument("--out", default="eval_output")
    p.add_argument("--num_correct", type=int, default=5,
                   help="Số ảnh đoán đúng mỗi lớp để vẽ Grad-CAM")
    p.add_argument("--max_wrong", type=int, default=60,
                   help="Số ảnh đoán sai tối đa để vẽ Grad-CAM (tất cả vẫn được copy)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--imagenet_norm", action="store_true",
                   help="Bật nếu lúc train có Normalize theo ImageNet (ResNet/EfficientNet pretrained)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- Dữ liệu ---
    df = pd.read_csv(args.csv).dropna(subset=["id", args.label_col]).reset_index(drop=True)
    classes = sorted(df[args.label_col].unique())          # cùng cách đánh số lúc train
    class_to_idx = {c: i for i, c in enumerate(classes)}
    if args.test_csv:
        test_df = pd.read_csv(args.test_csv).dropna(subset=["id", args.label_col])
    else:
        _, _, test_idx = split_train_val_test(df, args.label_col)
        test_df = df.iloc[test_idx]
    test_df = test_df.reset_index(drop=True)
    print(f"Lớp: {classes}")
    print(f"Số ảnh Test: {len(test_df)}")

    tfm_list = [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]
    if args.imagenet_norm:
        tfm_list.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    test_ds = ImageDataset(test_df, args.img_dir, class_to_idx, args.label_col,
                           transforms.Compose(tfm_list))
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # --- Mô hình ---
    model = build_model(args.model, len(classes))
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)

    # --- Dự đoán ---
    probs, y_true, idxs = predict_all(model, loader, device)
    y_pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)

    res = test_df.iloc[idxs].copy().reset_index(drop=True)
    res["true_idx"] = y_true
    res["pred_idx"] = y_pred
    res["true_label"] = [classes[i] for i in y_true]
    res["pred_label"] = [classes[i] for i in y_pred]
    res["confidence"] = conf
    res["correct"] = y_true == y_pred
    res[["id", "true_label", "pred_label", "confidence", "correct", "true_idx", "pred_idx"]] \
        .to_csv(out / "predictions.csv", index=False)

    # --- Chỉ số ---
    acc = res["correct"].mean()
    print(f"\nTest accuracy: {acc:.4f} ({res['correct'].sum()}/{len(res)})")

    labels_range = list(range(len(classes)))
    cm = confusion_matrix(y_true, y_pred, labels=labels_range)
    plot_confusion_matrix(cm, classes, out / "confusion_matrix.png")
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(out / "confusion_matrix.csv")

    report_txt = classification_report(y_true, y_pred, labels=labels_range,
                                       target_names=classes, digits=4, zero_division=0)
    report_dict = classification_report(y_true, y_pred, labels=labels_range,
                                        target_names=classes, output_dict=True, zero_division=0)
    (out / "classification_report.txt").write_text(report_txt, encoding="utf-8")
    pd.DataFrame(report_dict).T.to_csv(out / "classification_report.csv")
    print("\n" + report_txt)

    print("Các cặp bị nhầm nhiều nhất (thật -> đoán):")
    for n, t, pr in top_confusions(cm, classes):
        print(f"  {t} -> {pr}: {n} ảnh")

    if res["correct"].any() and (~res["correct"]).any():
        print(f"\nĐộ tự tin trung bình khi đoán đúng: {res.loc[res['correct'], 'confidence'].mean():.3f}")
        print(f"Độ tự tin trung bình khi đoán sai : {res.loc[~res['correct'], 'confidence'].mean():.3f}")

    # --- Gom ảnh đoán sai vào một tập riêng ---
    wrong = res[~res["correct"]].copy()
    wrong_dir = out / "misclassified"
    for _, r in wrong.iterrows():
        dst = wrong_dir / f"{r['true_label']}__as__{r['pred_label']}"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(args.img_dir) / f"{r['id']}.jpg", dst / f"{r['id']}.jpg")
    wrong.sort_values("confidence", ascending=False) \
        .to_csv(out / "misclassified.csv", index=False)
    print(f"\nĐã gom {len(wrong)} ảnh đoán sai vào: {wrong_dir}")

    # --- Grad-CAM ---
    cam_engine = GradCAM(model, get_target_layer(model, args.model))
    index_rows = []

    # Ảnh đoán sai: ưu tiên những ảnh mô hình tự tin nhất mà vẫn sai
    wrong_pick = wrong.sort_values("confidence", ascending=False).head(args.max_wrong)
    for _, r in wrong_pick.iterrows():
        paths = save_gradcam_case(cam_engine, r, args.img_dir, classes,
                                  out / "gradcam" / "wrong", args.imagenet_norm, device)
        index_rows.append({**r[["id", "true_label", "pred_label", "confidence", "correct"]].to_dict(), **paths})

    # Ảnh đoán đúng: lấy ngẫu nhiên num_correct ảnh mỗi lớp
    rng = np.random.default_rng(42)
    correct = res[res["correct"]]
    for cls in classes:
        sub = correct[correct["true_label"] == cls]
        if len(sub) == 0:
            continue
        pick = sub.iloc[rng.permutation(len(sub))[: args.num_correct]]
        for _, r in pick.iterrows():
            paths = save_gradcam_case(cam_engine, r, args.img_dir, classes,
                                      out / "gradcam" / "correct", args.imagenet_norm, device)
            index_rows.append({**r[["id", "true_label", "pred_label", "confidence", "correct"]].to_dict(), **paths})

    pd.DataFrame(index_rows).to_csv(out / "gradcam_index.csv", index=False)
    print(f"Đã vẽ Grad-CAM cho {len(index_rows)} ảnh: {out / 'gradcam'}")
    print(f"Chỉ mục để dựng giao diện: {out / 'gradcam_index.csv'}")


if __name__ == "__main__":
    main()
