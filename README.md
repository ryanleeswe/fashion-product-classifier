# Fashion Product Classifier

Deep Learning project for classifying fashion product images with PyTorch and Streamlit.

## Chạy web demo

```bash
pip install -r requirements.txt
streamlit run app.py
```

The target dataset contains 8,000 images across eight classes (1,000 per class): Casual Shoes, Handbags, Jeans, Shirts, Sports Shoes, Heels, Tshirts, and Watches. The demo loads a checkpoint and its matching CSV, then shows the top three predictions and Grad-CAM. It warns when the top two probabilities are within 10 percentage points. An optional panel displays saved evaluation results under `eval_output/upgrade_resnet18` (or `FASHION_EVAL_DIR`).

To run a trained checkpoint locally, set the model, checkpoint, and matching CSV before starting Streamlit (PowerShell example):

```powershell
$env:FASHION_MODEL = "resnet18"
$env:FASHION_MODEL_PATH = "checkpoints/resnet18/resnet18_best.pth"
$env:FASHION_CSV_PATH = "clean_dataset.csv"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Train a new checkpoint before using this dataset. Checkpoints trained with the previous four-class dataset are not compatible with the eight-class output layer.

## Tạo biểu đồ t-SNE

Run from the project root after the test images and checkpoint are available:

```bash
python visualize_tsne.py --model baseline --weights checkpoints/baseline/baseline_best.pth --csv clean_dataset.csv --img_dir training_images --imagenet_norm --out eval_output/baseline/tsne.png
```

By default, the script uses the same seed-42 stratified test split as `evaluate.py`, and plots up to 300 images per class. Use `--max_per_class 0` to include every test image. For a separately saved test CSV, pass `--test_csv test.csv`. Use `--imagenet_norm` if the selected model was trained with ImageNet normalization (as indicated by `evaluate.py`). The chart uses penultimate-layer features; t-SNE axes and cluster distances are not direct measures of model accuracy.

The chart includes all labels in the selected CSV. Keep the CSV, image directory, and checkpoint from the same training run.

## Huấn luyện và đánh giá

The CSV needs `id` and `articleType` columns, with each image stored as `<id>.jpg` under the image directory. Training and evaluation use a stratified 70/20/10 train/validation/test split by default, with seed 42.

```bash
python train.py --model baseline --csv clean_dataset.csv --image-dir training_images --epochs 10 --batch-size 32 --output-dir checkpoints/baseline
python evaluate.py --model baseline --weights checkpoints/baseline/baseline_best.pth --csv clean_dataset.csv --img_dir training_images --imagenet_norm --out eval_output/baseline
python visualize_tsne.py --model baseline --weights checkpoints/baseline/baseline_best.pth --csv clean_dataset.csv --img_dir training_images --imagenet_norm --out eval_output/baseline/tsne.png

python train.py --model improved --csv clean_dataset.csv --image-dir training_images --epochs 10 --batch-size 32 --output-dir checkpoints/improved
python evaluate.py --model improved --weights checkpoints/improved/improved_best.pth --csv clean_dataset.csv --img_dir training_images --imagenet_norm --out eval_output/improved
python visualize_tsne.py --model improved --weights checkpoints/improved/improved_best.pth --csv clean_dataset.csv --img_dir training_images --imagenet_norm --out eval_output/improved/tsne.png

python train.py --model resnet18 --csv clean_dataset.csv --image-dir training_images --epochs 10 --batch-size 32 --output-dir checkpoints/resnet18
python evaluate.py --model resnet18 --weights checkpoints/resnet18/resnet18_best.pth --csv clean_dataset.csv --img_dir training_images --imagenet_norm --out eval_output/resnet18
python visualize_tsne.py --model resnet18 --weights checkpoints/resnet18/resnet18_best.pth --csv clean_dataset.csv --img_dir training_images --imagenet_norm --out eval_output/resnet18/tsne.png
```

Evaluation and t-SNE use the same test split as training by default. Training applies ImageNet normalization, so include `--imagenet_norm` for evaluation and t-SNE. Try `--model improved` if you need a smaller model to train faster on CPU.

## Huấn luyện trên Google Colab

Open `colab_train_resnet18.ipynb` in Colab and choose **Runtime > Change runtime type > GPU**. Put `colab_code.zip`, `training_images.zip` (containing the `training_images/` folder), and `clean_dataset.csv` in `MyDrive/fashion-product-classifier`, mount Drive, then run the cells in order. The notebook saves the trained checkpoint and evaluation results back to Drive. Colab GPU availability and runtime limits vary; see the [official FAQ](https://research.google.com/colaboratory/faq.html).

## Chạy bằng Docker Compose

Docker đóng gói ứng dụng, thư viện Python và mã nguồn cần thiết. Ảnh training không cần cho màn hình dự đoán; checkpoint, CSV và kết quả đánh giá được mount read-only từ thư mục dự án:

```bash
docker compose up --build
```

Open <http://localhost:8501>. Stop with `Ctrl+C`, or run `docker compose down` in another terminal. Docker Desktop (or Docker Engine with the Compose plugin) must be installed. The container uses CPU PyTorch and does not require CUDA.

Train first with the command above. Compose reads `clean_dataset.csv` and the checkpoint in `checkpoints/`:

```text
FASHION_MODEL=resnet18
FASHION_MODEL_PATH=/app/checkpoints/resnet18/resnet18_best.pth
FASHION_CSV_PATH=/app/clean_dataset.csv
```
