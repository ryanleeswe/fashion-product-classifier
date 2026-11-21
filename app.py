from pathlib import Path
import os

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from models import create_model


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("FASHION_MODEL_PATH", str(PROJECT_DIR / "checkpoints" / "baseline" / "baseline_best.pth"))).expanduser()
MODEL_NAME = os.environ.get("FASHION_MODEL", "baseline").lower()
DATASET_PATH = Path(os.environ.get("FASHION_CSV_PATH", str(PROJECT_DIR / "clean_dataset.csv"))).expanduser()
EVAL_DIR = Path(
    os.environ.get("FASHION_EVAL_DIR", str(PROJECT_DIR / "eval_output" / "upgrade_resnet18"))
).expanduser()
IMAGE_SIZE = 128
TOP_K = 3
CONFUSION_MARGIN = 0.10  # Ngưỡng chênh Top-1/Top-2 là 10 phần trăm.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

st.set_page_config(
    page_title="Hệ thống phân loại thời trang và giải thích AI",
    page_icon="🛍️",
    layout="centered",
)

st.title("🛍️ Hệ thống Phân loại Sản phẩm Thời trang Thông minh")
st.markdown("Tích hợp Deep Learning, Transfer Learning và Giải thích AI (Grad-CAM).")


@st.cache_data
def load_class_names() -> list[str]:
    """Nạp danh sách lớp theo đúng thứ tự chữ cái đã dùng khi huấn luyện."""
    if DATASET_PATH.is_file():
        classes = (
            pd.read_csv(DATASET_PATH)["articleType"]
            .dropna()
            .astype(str)
            .sort_values()
            .unique()
            .tolist()
        )
        if len(classes) >= 2:
            return classes
    return [
        "Casual Shoes", "Handbags", "Shirts", "Sports Shoes",
        "Watches", "Heels", "Tshirts", "Jeans",
    ]


CLASS_NAMES = load_class_names()


@st.cache_resource
def load_trained_model() -> tuple[torch.nn.Module, torch.device]:
    """Nạp checkpoint của mô hình đã chọn cho ứng dụng web."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {MODEL_PATH.name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(MODEL_NAME, num_classes=len(CLASS_NAMES), pretrained=False)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


def prepare_image(image: Image.Image, device: torch.device) -> tuple[torch.Tensor, np.ndarray]:
    display_image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    image_array = np.asarray(display_image, dtype=np.float32) / 255.0
    normalized = (image_array - IMAGENET_MEAN) / IMAGENET_STD
    input_tensor = (
        torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).to(device)
    )
    return input_tensor, image_array


def make_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    class_index: int,
    image_array: np.ndarray,
    model_name: str,
) -> np.ndarray:
    """Tạo bản overlay Grad-CAM tại lớp tích chập cuối của mô hình."""
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    if model_name in ("baseline", "improved"):
        target_layer = model.features[3] if model_name == "baseline" else model.features[10]
    elif model_name in ("resnet18", "resnet18_se"):
        target_layer = model.layer4[-1]
    elif model_name == "efficientnet_b0":
        target_layer = model.features[-1]
    else:
        raise ValueError(f"Model chưa hỗ trợ: {model_name}")

    def save_activation(_module, _inputs, output):
        activations.append(output)
        output.register_hook(lambda gradient: gradients.append(gradient))

    hook = target_layer.register_forward_hook(save_activation)
    try:
        model.zero_grad()
        logits = model(input_tensor)
        logits[0, class_index].backward()
    finally:
        hook.remove()

    weights = gradients[0].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activations[0]).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(
        cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
    )[0, 0].detach().cpu().numpy()
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()

    from matplotlib import colormaps

    heatmap = colormaps["jet"](cam)[..., :3]
    return np.clip((0.55 * image_array) + (0.45 * heatmap), 0, 1)


try:
    model, device = load_trained_model()
except (FileNotFoundError, RuntimeError) as error:
    st.error(f"Chưa thể nạp model demo: {error}")
    st.stop()

uploaded_file = st.file_uploader(
    "Tải lên bức ảnh sản phẩm thời trang cần kiểm tra...",
    type=["jpg", "png", "jpeg"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Ảnh đầu vào")
        st.image(image, caption="Sản phẩm thực tế", use_container_width=True)

    if st.button("🚀 Chạy AI Phân tích & Giải thích", type="primary"):
        with st.spinner("AI đang xử lý và trích xuất đặc trưng..."):
            input_tensor, image_array = prepare_image(image, device)
            with torch.no_grad():
                probabilities = torch.softmax(model(input_tensor), dim=1)[0]
            confidence, prediction = torch.max(probabilities, 0)
            predicted_index = prediction.item()
            heatmap_image = make_gradcam(
                model, input_tensor, predicted_index, image_array, MODEL_NAME
            )

        confidence_score = confidence.item() * 100
        predicted_class = CLASS_NAMES[predicted_index]

        st.markdown("---")
        st.subheader("📊 Kết quả Phân tích từ AI")
        if confidence_score > 80:
            st.success(
                f"**Dự đoán:** {predicted_class} "
                f"(Độ tự tin: **{confidence_score:.2f}%**)"
            )
        else:
            st.warning(
                f"**Dự đoán:** {predicted_class} "
                f"(Độ tự tin thấp: **{confidence_score:.2f}%**)"
            )

        with col2:
            st.subheader("Bản đồ nhiệt Grad-CAM")
            st.image(
                heatmap_image,
                caption="Vùng AI tập trung nhìn (Màu đỏ/vàng)",
                use_container_width=True,
            )

        st.markdown("### 🔍 Phân tích chi tiết (Top 3 khả năng cao nhất):")
        top_count = min(TOP_K, len(CLASS_NAMES))
        top_probabilities, top_indices = torch.topk(probabilities, top_count)
        if top_count >= 2:
            top1_probability, top2_probability = top_probabilities[:2]
            margin = (top1_probability - top2_probability).item()
            if margin <= CONFUSION_MARGIN:
                first_class = CLASS_NAMES[top_indices[0].item()]
                second_class = CLASS_NAMES[top_indices[1].item()]
                st.warning(
                    f"⚠️ AI đang phân vân giữa **{first_class}** "
                    f"({top1_probability.item() * 100:.1f}%) và **{second_class}** "
                    f"({top2_probability.item() * 100:.1f}%) do độ chênh chỉ "
                    f"{margin * 100:.1f} điểm %. Vui lòng chuyển ảnh cho người kiểm tra."
                )
        for probability, class_index in zip(top_probabilities, top_indices):
            st.progress(
                int(probability.item() * 100),
                text=f"{CLASS_NAMES[class_index.item()]}: {probability.item() * 100:.2f}%",
            )


st.markdown("---")
with st.expander("📊 Đánh giá mô hình trên tập test", expanded=False):
    st.caption(
        "Các biểu đồ dưới đây là kết quả đánh giá offline trên tập test; "
        "chúng không thay đổi theo ảnh bạn vừa tải lên."
    )
    confusion_path = EVAL_DIR / "confusion_matrix.png"
    tsne_path = EVAL_DIR / "tsne.png"
    report_path = EVAL_DIR / "classification_report.csv"

    if not any(path.is_file() for path in (confusion_path, tsne_path, report_path)):
        st.info(
            f"Chưa tìm thấy kết quả đánh giá trong `{EVAL_DIR}`. "
            "Hãy chép confusion_matrix.png, tsne.png và "
            "classification_report.csv từ kết quả Colab vào thư mục này."
        )
    else:
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            if confusion_path.is_file():
                st.image(confusion_path, caption="Confusion matrix trên tập test", use_container_width=True)
        with chart_col2:
            if tsne_path.is_file():
                st.image(tsne_path, caption="t-SNE đặc trưng của ảnh test", use_container_width=True)
        if report_path.is_file():
            st.markdown("#### Chỉ số theo lớp")
            report = pd.read_csv(report_path, index_col=0)
            st.dataframe(report, use_container_width=True)
