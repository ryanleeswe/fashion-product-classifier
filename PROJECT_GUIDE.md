# Bản đồ dự án Fashion Product Classifier

Tài liệu này giúp thành viên mới hiểu luồng từ dữ liệu đến demo, biết mỗi nhóm file đóng góp gì và tìm đúng bằng chứng khi viết báo cáo. Chi tiết thuật toán nằm trong mã nguồn được liên kết.

## 1. Tóm tắt trong một phút

Dự án phân loại ảnh thời trang thành tám lớp: `Casual Shoes`, `Handbags`, `Jeans`, `Shirts`, `Sports Shoes`, `Heels`, `Tshirts` và `Watches`. Dataset hiện tại có 8.000 ảnh, 1.000 ảnh mỗi lớp. `train.py` chia dữ liệu stratified theo seed 42, huấn luyện model và chọn checkpoint theo validation accuracy.

```text
clean_dataset.csv + training_images/
        → train/validation/test
        → baseline / improved / ResNet18
        → checkpoint tốt nhất
        → đánh giá + confusion matrix + t-SNE
        → Streamlit: Top 3 + Grad-CAM
```

Kết quả test hiện có: ResNet18 đạt 95,00%, baseline 89,38%, improved 89,13% trên cùng tập test 800 ảnh. Đây là kết quả của split hiện tại, không đảm bảo cùng hiệu năng trên ảnh bên ngoài dataset.

## 2. Bản đồ file theo vai trò

### Dữ liệu và chuẩn bị

- `clean_dataset.csv`: nhãn hiện tại; các cột chính là `id` và `articleType`.
- `training_images/<id>.jpg`: ảnh khớp ID trong CSV; đây là cặp dữ liệu dùng trong các lệnh train hiện tại.
- `training_images_old/`: bộ ảnh cũ; không dùng cho kết quả hiện tại.
- `prepare_data.py`: lấy tám lớp, tối đa 1.000 ảnh hợp lệ mỗi lớp từ `styles.csv` và `images/`, rồi tạo `clean_dataset.csv` và `training_images/`.
- `resize_chia_argumentation_normalize.py`: luồng thử nghiệm tạo ảnh resize và CSV chia tập riêng. `train.py` hiện resize và augmentation trực tiếp lúc nạp ảnh, nên file này không cần chạy cho pipeline chính.

### Model và huấn luyện

- `baseline_cnn.py`: định nghĩa `BaselineCNN` và `ImprovedCNN`. Baseline có hai khối convolution rồi flatten. Improved có ba khối convolution, BatchNorm, LeakyReLU, dropout và pooling ra lưới 4×4 để giữ một phần thông tin vị trí.
- `models.py`: chọn một trong ba model hiện dùng: `baseline`, `improved` và `resnet18`.
- `train.py`: kiểm tra dữ liệu, chia train/validation/test, huấn luyện, chọn checkpoint theo validation và ghi metric. Các model trong script này dùng resize 128×128, augmentation khi train và ImageNet normalization.
- `train_val.py`: các vòng lặp train và validation được `train.py` gọi.
- `best.py` và `train_basic.py`: pipeline baseline cũ/độc lập; các lệnh trong quy trình hiện tại dùng `train.py`.
- `checkpoints/<model>/`: checkpoint cùng metadata và kết quả của từng model.

Mỗi lần huấn luyện lưu `<model>_best.pth` và `<model>_best.json`. Các bảng trong thư mục output gồm `results.csv`, `epoch_metrics.csv` và `test_details.csv`. Chạy từng model vào thư mục riêng để bảng kết quả không ghi đè nhau.

Improved hiện dùng `AdaptiveAvgPool2d((4, 4))`. Một số chuỗi mô tả kiến trúc trong `baseline_cnn.py`, `models.py` và metadata cũ vẫn ghi GAP; phần mô tả đó chưa được cập nhật. Cấu trúc chạy thực tế nằm trong lớp `ImprovedCNN`.

### Đánh giá và phân tích

- `evaluate.py`: đánh giá checkpoint, ghi accuracy, precision/recall/F1, confusion matrix, dự đoán và ảnh Grad-CAM.
- `visualize_tsne.py`: lấy đặc trưng ảnh test để vẽ t-SNE; biểu đồ giúp quan sát phân bố, không phải thước đo accuracy.
- `eval_output/baseline/`: nơi lưu báo cáo baseline được đánh giá với preprocessing ImageNet.
- `eval_output/improved/` và `eval_output/resnet18/`: báo cáo đánh giá của các model tương ứng.

### Ứng dụng

- `app.py`: nạp checkpoint và CSV nhãn, nhận ảnh upload, hiển thị Top 3, cảnh báo khi hai lớp đầu gần nhau, Grad-CAM và kết quả đánh giá offline.
- `README.md`: hướng dẫn tổng quát, Colab và chạy ứng dụng trên máy cá nhân; một số lệnh mẫu vẫn dùng dataset cũ.
- `PROJECT_GUIDE.md`: bản đồ vai trò và luồng làm việc (tài liệu này).

## 3. Các phần việc và cách bàn giao

- **Data:** đọc `prepare_data.py`, CSV và thư mục ảnh. Bàn giao số ảnh/lớp, cách ghép ID với ảnh, quy tắc làm sạch và giới hạn dữ liệu.
- **Model/train:** đọc `baseline_cnn.py`, `models.py`, `train.py` và `train_val.py`. Bàn giao kiến trúc, preprocessing, cấu hình train, log epoch và checkpoint được chọn.
- **Đánh giá:** đọc `evaluate.py` và `visualize_tsne.py`. Bàn giao test accuracy, chỉ số theo lớp, confusion matrix, t-SNE, ví dụ lỗi và cách diễn giải.
- **Demo:** đọc `app.py`. Bàn giao luồng sử dụng, ảnh chụp Top 3/Grad-CAM và giới hạn của các hiển thị.
- **Tổng hợp:** đối chiếu dataset, checkpoint, seed và split; không trộn số liệu từ lần chạy hoặc dataset khác.

Khi bàn giao, kèm theo **đầu ra**, **cách tạo đầu ra** và **ý nghĩa/giới hạn** của nó để người viết phần khác có thể kiểm tra kết quả.

## 4. Quy trình chạy hiện tại

Chạy từ thư mục gốc dự án. Dataset cần có `clean_dataset.csv` và ảnh `training_images/<id>.jpg`.

### Cài đặt

Nếu môi trường `.venv` chưa được tạo:

```powershell
py -m venv .venv
python -m pip install -r requirements.txt
```

### Huấn luyện

Mỗi model ghi vào thư mục riêng. Mặc định split là 70% train, 20% validation, 10% test, stratified với seed 42.

```powershell
python train.py --model baseline --csv clean_dataset.csv --image-dir training_images --epochs 10 --batch-size 32 --output-dir checkpoints/baseline
python train.py --model improved --csv clean_dataset.csv --image-dir training_images --epochs 10 --batch-size 32 --output-dir checkpoints/improved
python train.py --model resnet18 --csv clean_dataset.csv --image-dir training_images --epochs 10 --batch-size 32 --output-dir checkpoints/resnet18
```

`train.py` tự chọn CUDA nếu khả dụng. ResNet18 dùng pretrained ImageNet mặc định; baseline và improved khởi tạo trọng số mới. Checkpoint được chọn theo validation accuracy, sau đó test được dùng để báo cáo.

### Đánh giá và t-SNE

`train.py` chuẩn hóa đầu vào theo ImageNet cho cả ba model hiện tại. Vì vậy, khi đánh giá hoặc tạo t-SNE cho checkpoint từ script này, cần thêm `--imagenet_norm`.

```powershell
python evaluate.py --model baseline --weights checkpoints/baseline/baseline_best.pth --csv clean_dataset.csv --img_dir training_images --imagenet_norm --out eval_output/baseline
python evaluate.py --model improved --weights checkpoints/improved/improved_best.pth --csv clean_dataset.csv --img_dir training_images --imagenet_norm --out eval_output/improved
python evaluate.py --model resnet18 --weights checkpoints/resnet18/resnet18_best.pth --csv clean_dataset.csv --img_dir training_images --imagenet_norm --out eval_output/resnet18

python visualize_tsne.py --model baseline --weights checkpoints/baseline/baseline_best.pth --csv clean_dataset.csv --img_dir training_images --imagenet_norm --out eval_output/baseline/tsne.png
python visualize_tsne.py --model improved --weights checkpoints/improved/improved_best.pth --csv clean_dataset.csv --img_dir training_images --imagenet_norm --out eval_output/improved/tsne.png
python visualize_tsne.py --model resnet18 --weights checkpoints/resnet18/resnet18_best.pth --csv clean_dataset.csv --img_dir training_images --imagenet_norm --out eval_output/resnet18/tsne.png
```

Nếu báo cáo baseline hiện có được tạo trước khi dùng `--imagenet_norm`, hãy chạy lại lệnh baseline ở trên để cập nhật `eval_output/baseline/` trước khi so sánh.

### Chạy demo

Ví dụ chạy ResNet18:

```powershell
$env:FASHION_MODEL = "resnet18"
$env:FASHION_MODEL_PATH = "checkpoints/resnet18/resnet18_best.pth"
$env:FASHION_CSV_PATH = "clean_dataset.csv"
$env:FASHION_EVAL_DIR = "eval_output/resnet18"
python -m streamlit run app.py
```

CSV, thứ tự lớp và checkpoint phải thuộc cùng một dataset/lần train. Ứng dụng mặc định chọn baseline; đổi `FASHION_MODEL` và `FASHION_MODEL_PATH` khi dùng model khác.

## 5. Kết quả hiện tại

Các kết quả dưới đây lấy từ `checkpoints/<model>/results.csv` và `test_details.csv`. Mỗi model được đánh giá trên cùng split test 800 ảnh, 100 ảnh mỗi lớp.

| Model | Best validation accuracy | Test accuracy | Macro F1 | Tham số |
|---|---:|---:|---:|---:|
| Baseline | 89,75% | 89,38% (715/800) | 89,38% | 267.240 |
| Improved | 88,38% | 89,13% (713/800) | 89,02% | 110.088 |
| ResNet18 | 93,50% | 95,00% (760/800) | 95,02% | 11.180.616 |

ResNet18 hiện có test accuracy cao nhất trên split này. Improved gọn hơn baseline về số tham số nhưng kết quả hiện tại thấp hơn baseline 2 ảnh trên tập test; kiến trúc lớn hơn hoặc mới hơn không bảo đảm accuracy cao hơn.

Kết quả là phép đo trên một split/seed cố định, không đảm bảo hiệu năng tương tự với ảnh ngoài thực tế. Top 3 là xác suất softmax tương đối giữa các lớp, chưa được hiệu chỉnh như độ tin cậy. Grad-CAM minh họa vùng ảnh ảnh hưởng tới đầu ra, không chứng minh model hiểu sản phẩm.

## 6. Dàn ý báo cáo và nguồn bằng chứng

1. **Bài toán và dữ liệu:** tám lớp, 1.000 ảnh/lớp, ghép ID với ảnh, quy tắc làm sạch và giới hạn. Nguồn: `clean_dataset.csv`, `training_images/`, `prepare_data.py`.
2. **Phương pháp:** kiến trúc được chọn, resize 128×128, augmentation, ImageNet normalization, split stratified và seed. Nguồn: `baseline_cnn.py`, `models.py`, `train.py`.
3. **Kết quả:** validation/test accuracy, macro và per-class F1. Nguồn: `checkpoints/<model>/results.csv`, `test_details.csv`.
4. **Phân tích:** confusion matrix, các cặp lớp nhầm, t-SNE và ví dụ ảnh sai. Nguồn: `eval_output/<model>/confusion_matrix.png`, `tsne.png`, `misclassified.csv`.
5. **Demo và giới hạn:** ảnh upload, Top 3, Grad-CAM; nêu rõ tám lớp và kết quả đến từ split/seed cụ thể.

Trong báo cáo, ghi tên checkpoint, dataset, seed và split để người đọc đối chiếu. Không trộn metric của lần chạy hoặc dataset khác.

## 7. Lưu ý khi chia sẻ với nhóm

`.gitignore` bỏ qua checkpoint, kết quả đánh giá, ảnh và CSV/dữ liệu nặng. Người clone mã nguồn có thể không có `clean_dataset.csv`, `training_images/`, checkpoint hoặc ảnh báo cáo. Cần chia sẻ các đầu ra đó riêng hoặc thống nhất nơi lưu; không giả định chúng được commit cùng mã nguồn.
