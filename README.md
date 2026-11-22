# Fashion Product Classifier

Hệ thống phân loại ảnh sản phẩm thời trang bằng PyTorch và Streamlit. Dự án huấn luyện và so sánh BaselineCNN, ImprovedCNN và ResNet-18; sau đó đánh giá mô hình, tạo confusion matrix, t-SNE và Grad-CAM cho ứng dụng demo.

## Pipeline

```text
CSV nhãn + thư mục ảnh
        ↓
Kiểm tra dữ liệu và chia train/validation/test
        ↓
Resize 128×128 + normalization theo ImageNet
        ↓
Huấn luyện mô hình
        ↓
Chọn checkpoint tốt nhất theo validation accuracy
        ↓
Đánh giá test + confusion matrix + t-SNE
        ↓
Chạy demo Streamlit
```

## Cấu trúc thư mục

```text
fashion-product-classifier/
├── app.py                         # Demo Streamlit
├── requirements.txt
├── README.md
├── src/
│   ├── models/
│   │   ├── baseline_cnn.py        # BaselineCNN và ImprovedCNN
│   │   └── models.py              # Tạo mô hình
│   ├── training/
│   │   └── train.py               # Huấn luyện chính
│   └── evaluation/
│       ├── evaluate.py            # Đánh giá và Grad-CAM
│       └── visualize_tsne.py      # Trực quan t-SNE
├── scripts/
│   └── prepare_data.py            # Chuẩn bị dữ liệu
├── notebooks/                     # Notebook hỗ trợ
├── clean_dataset.csv              # Dữ liệu nhãn
├── training_images/               # Ảnh theo <id>.jpg
├── checkpoints/                   # Trọng số và metadata
└── eval_output/                   # Kết quả đánh giá
```

Do dung lượng, dữ liệu, ảnh, checkpoint và thư mục đánh giá có thể không có trong bản clone Git.

## Bắt đầu

### Bước 1: Clone dự án

```powershell
git clone https://github.com/ryanleeswe/fashion-product-classifier.git
cd fashion-product-classifier
```

### Bước 2: Tạo môi trường Python

Yêu cầu Python 3.10 trở lên:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu Windows chặn activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Bước 3: Chọn CPU hoặc GPU

#### Chạy bằng GPU NVIDIA

Kiểm tra driver hỗ trợ CUDA và cài phiên bản PyTorch tương thích. Ví dụ cho CUDA 12.8:

```powershell
python -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Kiểm tra GPU:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

#### Chạy bằng CPU

Không cần cài CUDA; script tự chuyển sang CPU khi không phát hiện GPU:

```powershell
python -m pip install torch torchvision
```

CPU chạy được, nhưng huấn luyện chậm hơn. Nên bắt đầu bằng batch size nhỏ.

### Bước 4: Chuẩn bị dữ liệu

CSV cần có hai cột:

- `id`: khớp tên file ảnh `<id>.jpg`.
- `articleType`: nhãn phân loại.

Ví dụ:

```text
clean_dataset.csv
training_images/
├── 10001.jpg
├── 10002.jpg
└── ...
```

Nếu đã có `styles.csv` và `images/`, tạo dữ liệu sạch bằng:

```powershell
python scripts/prepare_data.py
```

Mã huấn luyện sẽ loại dòng thiếu `id`/`articleType`, loại ID trùng và chia dữ liệu phân tầng theo nhãn với seed `42`:

- 70% train: cập nhật tham số mô hình.
- 20% validation: chọn checkpoint tốt nhất.
- 10% test: đánh giá sau khi huấn luyện.

### Bước 5: Huấn luyện

Chạy từng mô hình bằng GPU hoặc CPU:

```powershell
python -m src.training.train --model baseline --csv clean_dataset.csv --image-dir training_images --epochs 10 --batch-size 32 --output-dir checkpoints/baseline

python -m src.training.train --model improved --csv clean_dataset.csv --image-dir training_images --epochs 10 --batch-size 32 --output-dir checkpoints/improved

python -m src.training.train --model resnet18 --csv clean_dataset.csv --image-dir training_images --epochs 10 --batch-size 32 --output-dir checkpoints/resnet18
```

Hoặc chạy tất cả mô hình:

```powershell
python -m src.training.train --model all --csv clean_dataset.csv --image-dir training_images --epochs 10 --batch-size 32
```

Cấu hình chính:

- Ảnh RGB 128×128.
- Normalize theo mean/std của ImageNet.
- Cross-Entropy Loss và AdamW.
- Learning rate `1e-3`, weight decay `1e-4`.
- Dropout `0.3`.
- Tập train dùng lật ngang ngẫu nhiên `p=0.5` và xoay `±10°`.
- Validation/test không dùng augmentation.
- ResNet-18 pretrained: backbone dùng learning rate bằng 10% learning rate của head.
- Checkpoint được lưu tại epoch có validation accuracy cao nhất.

### Bước 6: Đánh giá

Ví dụ đánh giá ResNet-18:

```powershell
python -m src.evaluation.evaluate --model resnet18 --weights checkpoints/resnet18/resnet18_best.pth --csv clean_dataset.csv --img_dir training_images --imagenet_norm --out eval_output/resnet18
```

Kết quả gồm accuracy, precision, recall, F1 theo từng lớp, confusion matrix, danh sách dự đoán sai và Grad-CAM.

Tạo t-SNE:

```powershell
python -m src.evaluation.visualize_tsne --model resnet18 --weights checkpoints/resnet18/resnet18_best.pth --csv clean_dataset.csv --img_dir training_images --imagenet_norm --out eval_output/resnet18/tsne.png
```

Script lấy tối đa 300 ảnh mỗi lớp. Dùng `--max_per_class 0` để lấy toàn bộ tập test.

### Bước 7: Chạy demo

```powershell
$env:FASHION_MODEL = "resnet18"
$env:FASHION_MODEL_PATH = "checkpoints/resnet18/resnet18_best.pth"
$env:FASHION_CSV_PATH = "clean_dataset.csv"
$env:FASHION_EVAL_DIR = "eval_output/resnet18"
python -m streamlit run app.py
```

Mở địa chỉ Streamlit hiển thị trong terminal, thường là `http://localhost:8501`.

Ứng dụng hiển thị ảnh tải lên, Top 3 xác suất, Grad-CAM và cảnh báo nếu hai lớp hàng đầu quá gần nhau.

## Kiến trúc mô hình

| Mô hình | Kiến trúc |
|---|---|
| BaselineCNN | Hai khối Conv2D 3×3 với 16 và 32 kênh, ReLU, max pooling, flatten và Linear |
| ImprovedCNN | Ba khối Conv2D 3×3 với 32, 64 và 128 kênh, BatchNorm, LeakyReLU và AdaptiveAvgPool 4×4 |
| ResNet-18 | Backbone pretrained trên ImageNet, thay fully connected cuối bằng Dropout và Linear theo số lớp |

## Kết quả

Đọc số liệu trực tiếp từ:

- `checkpoints/<model>/results.csv`
- `checkpoints/<model>/epoch_metrics.csv`
- `checkpoints/<model>/test_details.csv`

Khi trình bày kết quả, ghi rõ dataset, checkpoint, seed, split và phần cứng đã dùng. Không trộn metric của các lần huấn luyện khác nhau.

Accuracy và F1 đánh giá mô hình trên tập test. t-SNE hỗ trợ quan sát không gian đặc trưng nhưng không đo trực tiếp độ chính xác. Softmax và Grad-CAM không thay thế kiểm tra thủ công.
