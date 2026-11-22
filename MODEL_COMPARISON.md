# Benchmark và hướng phát triển mô hình

`train.py` dùng một bộ chia train/validation/test cố định và cùng tiền xử lý cho mọi mô hình. Validation chọn checkpoint tốt nhất; test chỉ được chấm sau khi huấn luyện xong để ước lượng khả năng tổng quát hóa.

`train_cnn.ipynb` giữ quy trình trực quan kiểu notebook ban đầu và chỉ chạy một model mỗi lượt. Đổi `MODEL_NAME` trong cell cấu hình, rồi chạy lại các cell theo thứ tự. Notebook dùng train/validation 80/20, vẽ loss/accuracy và cập nhật `checkpoints/notebook_results.csv`; dùng `train.py --model all` nếu cần benchmark chung có tập test.

## Các mô hình

| Tên truyền vào | Cấu trúc | Khởi tạo |
|---|---|---|
| `baseline` | BaselineCNN hiện có: Conv/ReLU/MaxPool rồi Linear | Ngẫu nhiên |
| `improved` | ImprovedCNN hiện có: BatchNorm, LeakyReLU, Dropout, AdaptiveAvgPool 4×4 | Ngẫu nhiên |
| `resnet18` | ResNet-18, thay lớp phân loại cuối bằng Dropout và Linear | ImageNet pretrained mặc định |

`models.py:create_model()` dựng một trong ba mô hình được dùng trong báo cáo. ResNet18 dùng transfer learning: thay đầu phân loại ImageNet bằng đầu ra tương ứng với số lớp của bộ dữ liệu. Có thể tắt pretrained bằng `--no-pretrained`.

## Mô hình có kết quả tốt nhất: ResNet18

ResNet18 được chọn để so sánh với hai CNN tự xây dựng vì mô hình có kết nối residual và có thể khởi tạo bằng trọng số ImageNet. Trong lần đánh giá được ghi nhận hiện tại, ResNet18 đạt kết quả cao nhất trong ba cấu hình đã chạy. Đây là kết luận cho các checkpoint và tập test hiện có, không khẳng định mọi cấu hình ResNet18 luôn vượt các CNN khác.

Bảng kết quả ghi số tham số, thời gian mỗi epoch, thời gian tới epoch validation tốt nhất, validation score và test score. `test_details.csv` có precision/recall/F1 theo từng lớp và confusion matrix để nhận ra cặp nhãn thường bị nhầm.

## Dữ liệu khó hơn

Dataset hiện tại có tám lớp cân bằng, mỗi lớp 1.000 ảnh. `make_benchmark_dataset.py` là tiện ích độc lập, không sửa `prepare_data.py`: chạy để liệt kê articleType trong `styles.csv`, rồi chọn một tập con các lớp gần nhau để so sánh. Mọi mô hình phải dùng cùng CSV benchmark và cùng ảnh.

## Cách chạy (PowerShell)

```powershell
# Liệt kê nhãn và số ảnh trong bộ nguồn
python make_benchmark_dataset.py --csv styles.csv --image-dir images

# Thay tên ví dụ bằng các nhãn gần nhau có thật trong danh sách trên
python make_benchmark_dataset.py --csv styles.csv --image-dir images `
  --categories "Shirts,Tshirts,Heels,Sweatshirts" --samples-per-category 1000

# So sánh mọi cấu hình trên cùng split
python train.py --model all --csv benchmark_dataset.csv `
  --image-dir benchmark_images --epochs 10 --batch-size 32

# Chạy một mô hình riêng
python train.py --model resnet18 --csv benchmark_dataset.csv `
  --image-dir benchmark_images --epochs 10
```

Lần đầu chạy pretrained cần tải trọng số ImageNet qua mạng. Dùng `--no-pretrained` nếu muốn chạy không tải hoặc so sánh kiến trúc khởi tạo ngẫu nhiên. Kết quả nằm trong `checkpoints/`: `results.csv`, `epoch_metrics.csv`, `test_details.csv` và từng checkpoint `*_best.pth`. Thời gian benchmark phụ thuộc CPU/GPU, batch size và số ảnh; không nên so sánh tốc độ giữa các máy khác nhau.

## Tài liệu kiến trúc

- [TorchVision ResNet](https://docs.pytorch.org/vision/stable/models/resnet)
- [PyTorch transfer learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
