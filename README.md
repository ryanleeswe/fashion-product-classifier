# Fashion Product Classifier

Deep Learning project for classifying fashion product images with PyTorch and Streamlit.

## Chạy web demo

```bash
pip install -r requirements.txt
streamlit run app.py
```

The demo loads a checkpoint and its matching CSV. It shows the top three predictions, Grad-CAM, and a human-review warning when the top two probabilities are within 10 percentage points. An optional test-results panel displays the confusion matrix, t-SNE plot, and per-class report when those files are available under `eval_output/upgrade_resnet18` (or `FASHION_EVAL_DIR`). Adjust `CONFUSION_MARGIN` in `app.py` to change the warning threshold. This is a confidence-gap heuristic, not calibrated uncertainty.

To run a trained checkpoint locally, set the model, checkpoint, and matching CSV before starting Streamlit (PowerShell example):

```powershell
$env:FASHION_MODEL = "resnet18"
$env:FASHION_MODEL_PATH = "checkpoints/resnet18_best.pth"
$env:FASHION_CSV_PATH = "clean_dataset.csv"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Tạo biểu đồ t-SNE

Run from the project root after the test images and checkpoint are available:

```bash
python visualize_tsne.py --model baseline --weights baseline_cnn.pth --csv clean_dataset.csv --img_dir training_images --out eval_output/tsne.png
```

By default, the script uses the same seed-42 stratified test split as `evaluate.py`, and plots up to 300 images per class. Use `--max_per_class 0` to include every test image. For a separately saved test CSV, pass `--test_csv test.csv`. Use `--imagenet_norm` if the selected model was trained with ImageNet normalization (as indicated by `evaluate.py`). The chart uses penultimate-layer features; t-SNE axes and cluster distances are not direct measures of model accuracy.

The project currently contains four classes in its demo dataset. The chart includes all labels present in the selected CSV; it will show eight classes only when the CSV, images, and checkpoint were trained for those same eight classes.

## Huấn luyện và đánh giá

The CSV needs `id` and `articleType` columns, with each image stored as `<id>.jpg` under the image directory. Training and evaluation use a stratified 70/20/10 train/validation/test split by default, with seed 42.

```bash
python train.py --model resnet18 --csv clean_dataset.csv --image-dir training_images --epochs 10 --batch-size 32 --output-dir checkpoints
python evaluate.py --model resnet18 --weights checkpoints/resnet18_best.pth --csv clean_dataset.csv --img_dir training_images --imagenet_norm --out eval_output/resnet18
python visualize_tsne.py --model resnet18 --weights checkpoints/resnet18_best.pth --csv clean_dataset.csv --img_dir training_images --imagenet_norm --out eval_output/resnet18/tsne.png
```

Evaluation and t-SNE use the same test split as training by default. Training applies ImageNet normalization, so include `--imagenet_norm` for evaluation and t-SNE. Try `--model improved` if you need a smaller model to train faster on CPU.

## Huấn luyện trên Google Colab

Open `colab_train_resnet18.ipynb` in Colab and choose **Runtime > Change runtime type > GPU**. Put `colab_code.zip`, `anh_tong.zip`, and `datasets/upgrade_dataset.csv` in `MyDrive/fashion-product-classifier`, mount Drive in the notebook, then run the cells in order. The notebook saves the trained checkpoint and evaluation results back to Drive. Colab GPU availability and runtime limits vary; see the [official FAQ](https://research.google.com/colaboratory/faq.html).
