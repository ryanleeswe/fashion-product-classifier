# Benchmark và hướng phát triển mô hình

`train.py` dùng một bộ chia train/validation/test cố định và cùng tiền xử lý cho mọi mô hình. Validation chọn checkpoint tốt nhất; test chỉ được chấm sau khi huấn luyện xong để ước lượng khả năng tổng quát hóa.

`train_cnn.ipynb` giữ quy trình trực quan kiểu notebook ban đầu và chỉ chạy một model mỗi lượt. Đổi `MODEL_NAME` trong cell cấu hình, rồi chạy lại các cell theo thứ tự. Notebook dùng train/validation 80/20, vẽ loss/accuracy và cập nhật `checkpoints/notebook_results.csv`; dùng `train.py --model all` nếu cần benchmark chung có tập test.

## Các mô hình

| Tên truyền vào | Cấu trúc | Khởi tạo |
|---|---|---|
| `baseline` | BaselineCNN hiện có: Conv/ReLU/MaxPool rồi Linear | Ngẫu nhiên |
| `improved` | ImprovedCNN hiện có: BatchNorm, LeakyReLU, Dropout, Global Average Pooling | Ngẫu nhiên |
| `resnet18` | ResNet-18, thay lớp phân loại cuối bằng Dropout và Linear | ImageNet pretrained mặc định |
| `efficientnet_b0` | EfficientNet-B0, thay classifier cuối bằng Dropout và Linear | ImageNet pretrained mặc định |
| `resnet18_se` | ResNet-18 pretrained, thêm attention theo kênh ở feature map cuối và classifier có Dropout | ImageNet pretrained mặc định |

`models.py:create_model()` là hàm dựng mô hình theo tên; `train.py` không cần lặp code khởi tạo từng mạng. ResNet-18 và EfficientNet-B0 dùng transfer learning: thay đầu phân loại ImageNet bằng đầu ra đúng số lớp của bộ dữ liệu. Có thể tắt pretrained bằng `--no-pretrained`.

## Mô hình được phát triển: ResNet18-SE

Chọn ResNet-18 làm nhánh phát triển vì residual connection cho phép truyền đặc trưng qua các khối sâu, đồng thời ResNet-18 có kích thước phù hợp để fine-tune trên máy cá nhân. `resnet18_se` giữ backbone pretrained, rồi áp dụng attention kiểu Squeeze-and-Excitation lên feature map cuối: global pooling tóm tắt từng kênh, một bottleneck nhỏ tạo trọng số kênh, sau đó các kênh được tái cân bằng trước khi phân loại. Như vậy mạng có thể tăng/giảm mức chú ý tới đặc trưng hữu ích cho từng loại sản phẩm. Backbone dùng learning rate nhỏ hơn classifier và attention head để hạn chế làm hỏng đặc trưng đã học.

Đây là giả thuyết cần kiểm chứng bằng kết quả; không mặc định ResNet hay EfficientNet luôn chính xác hoặc nhanh hơn. Bảng kết quả ghi số tham số, thời gian mỗi epoch, thời gian tới epoch validation tốt nhất, validation score và test score. `test_details.csv` có precision/recall/F1 theo từng lớp và confusion matrix để nhận ra cặp nhãn thường bị nhầm.

## Dữ liệu khó hơn

Bộ hiện tại có bốn nhóm khá khác nhau (`Tshirts`, `Casual Shoes`, `Watches`, `Sunglasses`), nên chưa đủ để kết luận mô hình xử lý tốt các nhóm dễ nhầm. `make_benchmark_dataset.py` là tiện ích độc lập, không sửa `prepare_data.py`: chạy để liệt kê articleType trong `styles.csv`, rồi chọn các lớp gần nhau theo danh sách thật trong CSV. Chẳng hạn có thể xem xét nhóm áo gần nhau nếu các tên đó tồn tại trong tập dữ liệu. Mọi mô hình phải dùng cùng CSV benchmark và cùng ảnh.

## Cách chạy (PowerShell)

```powershell
# Liệt kê nhãn và số ảnh trong bộ nguồn
python make_benchmark_dataset.py --csv styles.csv --image-dir images

# Thay tên ví dụ bằng các nhãn gần nhau có thật trong danh sách trên
python make_benchmark_dataset.py --csv styles.csv --image-dir images `
  --categories "Shirts,Tshirts,Tops,Sweatshirts" --samples-per-category 1000

# So sánh mọi cấu hình trên cùng split
python train.py --model all --csv benchmark_dataset.csv `
  --image-dir benchmark_images --epochs 10 --batch-size 32

# Chạy một mô hình riêng
python train.py --model resnet18_se --csv benchmark_dataset.csv `
  --image-dir benchmark_images --epochs 10
```

Lần đầu chạy pretrained cần tải trọng số ImageNet qua mạng. Dùng `--no-pretrained` nếu muốn chạy không tải hoặc so sánh kiến trúc khởi tạo ngẫu nhiên. Kết quả nằm trong `checkpoints/`: `results.csv`, `epoch_metrics.csv`, `test_details.csv` và từng checkpoint `*_best.pth`. Thời gian benchmark phụ thuộc CPU/GPU, batch size và số ảnh; không nên so sánh tốc độ giữa các máy khác nhau.

## Tài liệu kiến trúc

- [TorchVision ResNet](https://docs.pytorch.org/vision/stable/models/resnet)
- [TorchVision EfficientNet](https://docs.pytorch.org/vision/stable/models/efficientnet)
- [PyTorch transfer learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
