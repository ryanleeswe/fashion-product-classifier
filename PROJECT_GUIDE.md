# Bản đồ dự án Fashion Product Classifier

Tài liệu này giúp thành viên mới hiểu luồng từ dữ liệu đến demo, biết mỗi nhóm file đóng góp gì và tìm đúng bằng chứng khi viết báo cáo. Chi tiết thuật toán nằm trong mã nguồn được liên kết.

## 1. Tóm tắt trong một phút

Dự án nhận ảnh sản phẩm thời trang và dự đoán một trong bốn lớp: `Casual Shoes`, `Shirts`, `Sports Shoes`, `Tshirts`. Dataset hiện dùng có 8.032 ảnh, cân bằng với 2.008 ảnh mỗi lớp. ResNet-18 pretrained trên ImageNet được fine-tune bằng PyTorch. Checkpoint tốt nhất đạt validation accuracy 93,71% và test accuracy 93,16% trên 804 ảnh. Ứng dụng Streamlit nhận ảnh mới, hiển thị Top 3 và Grad-CAM.

```text
CSV nhãn + ảnh → chia train/validation/test → ResNet-18 fine-tune
      → checkpoint tốt nhất → đánh giá test + t-SNE
      → Streamlit: Top 3 + Grad-CAM + kết quả đánh giá
```

## 2. Bản đồ file theo vai trò

### Dữ liệu và chuẩn bị

- `styles.csv`: metadata nguồn ban đầu.
- `images/`: ảnh nguồn theo ID.
- `prepare_data.py`: script chuẩn bị dataset cũ, tạo `clean_dataset.csv` và `training_images/`.
- `datasets/upgrade_dataset.csv`: nhãn dùng cho lần train hiện tại; các cột chính là `id` và `articleType`.
- `datasets/anh_tong/`: ảnh dùng cho lần train hiện tại, tên `<id>.jpg` khớp với CSV.

**Lưu ý:** `clean_dataset.csv` + `training_images/` là dữ liệu của luồng cũ/mẫu. Kết quả báo cáo hiện tại dùng `datasets/upgrade_dataset.csv` + `datasets/anh_tong/`. Khi chạy lại, phải truyền đúng cặp CSV và thư mục ảnh.

### Model và huấn luyện

- `models.py`: định nghĩa kiến trúc. Với `--model resnet18`, lấy ResNet-18 pretrained ImageNet rồi thay lớp phân loại cuối cho số lớp của dự án.
- `train.py`: xác thực dữ liệu, chia tập có stratify, huấn luyện, chọn checkpoint theo validation và lưu các chỉ số.
- `train_val.py`: các vòng lặp train một epoch và validation được `train.py` gọi.
- `checkpoints/resnet18_best.pth`: trọng số model tốt nhất; được dùng trong đánh giá và app.
- `checkpoints/epoch_metrics.csv`: loss và accuracy theo epoch.
- `checkpoints/results.csv`, `checkpoints/test_details.csv`: kết quả tổng hợp và dự đoán chi tiết.

Pretrained nghĩa là model đã học đặc trưng ảnh tổng quát từ ImageNet trước khi fine-tune. Model chưa biết bốn lớp thời trang của dự án cho đến khi train trên dataset này.

### Đánh giá và phân tích

- `evaluate.py`: đánh giá checkpoint trên tập test, ghi accuracy, precision/recall/F1, confusion matrix và ảnh dự đoán sai.
- `visualize_tsne.py`: lấy đặc trưng của ảnh test và chiếu xuống 2D để quan sát phân bố theo lớp.
- `eval_output/upgrade_resnet18/confusion_matrix.png`: xem các lớp model hay nhầm.
- `eval_output/upgrade_resnet18/classification_report.csv`: chỉ số precision, recall, F1 và support theo lớp.
- `eval_output/upgrade_resnet18/tsne.png`: biểu đồ đặc trưng; không phải thước đo accuracy.
- `eval_output/upgrade_resnet18/predictions.csv`, `misclassified.csv`: kiểm tra dự đoán để phân tích lỗi.

### Ứng dụng

- `app.py`: nạp checkpoint và CSV nhãn, nhận ảnh upload, hiển thị Top 3, cảnh báo khi hai lớp đầu gần nhau, Grad-CAM và kết quả đánh giá offline.
- `README.md`: hướng dẫn tổng quát, Colab và Docker; một số lệnh mẫu vẫn dùng dataset cũ.
- `PROJECT_GUIDE.md`: bản đồ vai trò và luồng làm việc (tài liệu này).

## 3. Các phần việc và cách bàn giao

- **Data:** đọc `prepare_data.py`, CSV và thư mục ảnh. Bàn giao số lượng ảnh/lớp, cách ghép ảnh với nhãn, quy tắc làm sạch, ví dụ ảnh và giới hạn dữ liệu.
- **Model/train:** đọc `models.py`, `train.py`, `train_val.py`. Bàn giao kiến trúc, ý nghĩa pretrained, cấu hình train, log epoch và checkpoint được chọn.
- **Đánh giá:** đọc `evaluate.py`, `visualize_tsne.py`. Bàn giao test accuracy, chỉ số theo lớp, confusion matrix, t-SNE, ví dụ lỗi và cách diễn giải.
- **Demo:** đọc `app.py`. Bàn giao luồng sử dụng, ảnh chụp Top 3/Grad-CAM và giới hạn của các hiển thị.
- **Tổng hợp:** đối chiếu dataset, checkpoint, seed và split giữa các phần; không trộn số liệu từ lần chạy hoặc dataset khác.

Khi một phần được bàn giao, nên kèm theo **đầu ra**, **cách tạo đầu ra** và **ý nghĩa/giới hạn** của nó. Như vậy người viết phần khác có thể trích dẫn kết quả mà không cần suy đoán.

## 4. Quy trình chạy hiện tại

Chạy từ thư mục gốc project trong PowerShell. Máy đã xác nhận: Windows, Python 3.14, PyTorch 2.15.0+cu132, NVIDIA RTX 4050 Laptop GPU 6 GB VRAM, RAM hệ thống 8 GB. Bắt đầu batch size 8.

### Cài đặt và kiểm tra CUDA

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu132
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Lệnh CUDA trên là cấu hình đã chạy thành công với Python 3.14 trên máy này. Máy khác cần chọn bản tương thích tại [trang cài PyTorch](https://pytorch.org/get-started/locally/).

### Train

```powershell
.\.venv\Scripts\python.exe train.py --model resnet18 --csv datasets/upgrade_dataset.csv --image-dir datasets/anh_tong --epochs 10 --batch-size 8 --output-dir checkpoints
```

Script tự chọn CUDA nếu khả dụng, chia dữ liệu stratified theo seed 42, lưu checkpoint có validation accuracy cao nhất và in kết quả test.

### Đánh giá và t-SNE

```powershell
.\.venv\Scripts\python.exe evaluate.py --model resnet18 --weights checkpoints/resnet18_best.pth --csv datasets/upgrade_dataset.csv --img_dir datasets/anh_tong --imagenet_norm --batch_size 8 --num_correct 0 --max_wrong 0 --out eval_output/upgrade_resnet18
.\.venv\Scripts\python.exe visualize_tsne.py --model resnet18 --weights checkpoints/resnet18_best.pth --csv datasets/upgrade_dataset.csv --img_dir datasets/anh_tong --imagenet_norm --batch_size 8 --out eval_output/upgrade_resnet18/tsne.png
```

`evaluate.py` tạo confusion matrix và báo cáo lớp; `visualize_tsne.py` tạo riêng `tsne.png`. Cả hai dùng cùng seed/split test như train.

### Chạy demo

```powershell
$env:FASHION_MODEL = "resnet18"
$env:FASHION_MODEL_PATH = "checkpoints/resnet18_best.pth"
$env:FASHION_CSV_PATH = "datasets/upgrade_dataset.csv"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Tải ảnh lên, bấm nút phân tích và cuộn xuống để xem Top 3. Mục **Đánh giá mô hình trên tập test** là kết quả offline cố định, không đổi theo ảnh upload.

## 5. Kết quả hiện tại

- 8.032 ảnh, 2.008 ảnh/lớp; split: 5.621 train / 1.607 validation / 804 test (xấp xỉ 70/20/10, seed 42).
- ResNet-18 pretrained ImageNet, fine-tune 10 epoch; checkpoint tốt nhất ở epoch 5.
- Validation accuracy **93,71%**; test accuracy **93,16% (749/804)**; macro F1 **93,15%**.
- F1 theo lớp: Casual Shoes **88,29%**, Shirts **98,26%**, Sports Shoes **87,53%**, Tshirts **98,51%**.
- Nhầm lẫn lớn nhất: Sports Shoes → Casual Shoes (28 ảnh), Casual Shoes → Sports Shoes (20 ảnh).

Các số này mô tả hiệu năng trên split test hiện tại, không đảm bảo kết quả tương tự với mọi ảnh ngoài thực tế. Top 3 là xác suất softmax tương đối giữa bốn lớp, chưa được hiệu chỉnh như độ tin cậy. Grad-CAM minh họa vùng ảnh ảnh hưởng tới đầu ra, không chứng minh model hiểu sản phẩm. t-SNE giúp quan sát đặc trưng; khoảng cách trên hình không phải accuracy.

## 6. Dàn ý báo cáo và nguồn bằng chứng

1. **Bài toán và dữ liệu:** bốn lớp, số ảnh/lớp, ghép ID với ảnh, làm sạch và giới hạn. Nguồn: CSV, `datasets/anh_tong/`.
2. **Phương pháp:** ResNet-18 pretrained, fine-tune, resize 128×128, ImageNet normalization, augmentation, split stratified và seed. Nguồn: `models.py`, `train.py`, log train.
3. **Kết quả:** validation/test accuracy, macro và per-class F1. Nguồn: log train, `epoch_metrics.csv`, `classification_report.csv`.
4. **Phân tích:** confusion matrix, các cặp lớp nhầm, t-SNE và ví dụ ảnh sai. Nguồn: `confusion_matrix.png`, `tsne.png`, `misclassified.csv`.
5. **Demo và giới hạn:** ảnh upload, Top 3, Grad-CAM; nêu rõ chỉ có bốn lớp và kết quả hiện tại đến từ một split/seed.

Trong báo cáo, ghi tên checkpoint, dataset, seed và split để người đọc đối chiếu. Không trộn metric từ lần chạy khác hoặc dataset cũ.

## 7. Lưu ý khi chia sẻ với nhóm

`.gitignore` đang loại `datasets/`, `checkpoints/` và `eval_output/` khỏi Git. Người clone source có thể không nhận được dữ liệu, checkpoint fine-tune hoặc hình báo cáo. Nhóm cần chia sẻ chúng riêng hoặc thống nhất nơi lưu; không giả định chúng được commit cùng mã nguồn. Trọng số pretrained ImageNet có thể tải lại, còn dataset và checkpoint của dự án cần được bảo quản.
