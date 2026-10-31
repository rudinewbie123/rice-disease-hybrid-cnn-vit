# Rice Disease Classification: CNN, ViT, and Hybrid CNN–ViT

Research implementation for **Pengembangan Model Prediksi Penyakit Tanaman Padi Berbasis Citra Daun Menggunakan Hybrid CNN–Vision Transformer**.

Three configured classes: bacterial leaf blight, brown spot, and leaf smut. Architectures: EfficientNet-B0 CNN, torchvision ViT-B/16, and Hybrid CNN–ViT (EfficientNet spatial encoder + transformer token encoder).

**No dataset, trained checkpoint, synthetic image, or experimental result is included.** Raw images remain in Google Drive. CSV manifests use columns `image_path,label`; paths may be absolute or relative to `data_root`. Required files: `splits/train.csv`, `splits/val.csv`, `splits/test.csv`.

Install: `pip install -e '.[dev]'`. Run `python scripts/train.py --config configs/default.yaml --model efficientnet_b0` (also `vit_b16`, `hybrid`), then evaluate the held-out test split using `python scripts/evaluate.py --checkpoint PATH --config configs/default.yaml --split test`.

Evaluation reports accuracy, macro precision/recall/F1, classification report, confusion matrix, and macro one-vs-rest ROC-AUC when class support permits. Explainability includes Grad-CAM and actual multi-head transformer attention rollout for the hybrid model. Use identical data splits and training protocol for all model comparisons; never tune against the test split. See the Colab notebook in `notebooks/`.
