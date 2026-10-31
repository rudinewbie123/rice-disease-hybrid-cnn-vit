# Rice Disease Classification: CNN, ViT, and Hybrid CNN–ViT

Research implementation for **Pengembangan Model Prediksi Penyakit Tanaman Padi Berbasis Citra Daun Menggunakan Hybrid CNN–Vision Transformer**.

## Dataset provenance

The selected reference is Rahmi21's Kaggle notebook: [Klasifikasi Penyakit Padi Melalui Citra Daun](https://www.kaggle.com/code/rahmi21/klasifikasi-penyakit-padi-melalui-citra-daun). The notebook lists two inputs: **Classification of Rice Diseases Using Leaf Images** and **Data Ekstraksi**. This project deliberately selects only the former; the auxiliary feature-extraction input and field-collected samples are not merged.

The associated Rahmi & Saputra (2024) publication identifies three target classes: **blast**, **blight**, and **tungro**. The published study describes 240 Kaggle-sourced images plus 260 field-collected images in its combined study data; those combined counts must not be interpreted as the count of the Kaggle dataset alone. This repository does not hard-code a dataset size. Actual image counts are generated from the downloaded source by the audit utility.

Dataset download requires the user's Kaggle credentials and the exact dataset slug shown by Kaggle. Raw images remain in Google Drive and are never committed to GitHub. Do not add the auxiliary "Data Ekstraksi" input or the field-collected images to this source-only experiment.

## Google Drive layout

Place the extracted Kaggle dataset inside the Drive data root, preserving its class-folder structure:

```text
rice_disease_dataset/
  raw/
    Classification of Rice Diseases Using Leaf Images/
      Blast/
      Blight/
      Tungro/
  splits/
    train.csv
    val.csv
    test.csv
    dataset_audit.json
```

Folder names may use common aliases such as `Leaf Blast`, `Bacterial Leaf Blight`, and `Rice Tungro`. The preparation script recognizes these aliases. If the Kaggle archive uses different names, inspect them and update `CLASS_ALIASES` explicitly rather than guessing.

## Prepare the dataset

After downloading and extracting the Kaggle dataset into Google Drive, run:

```bash
python scripts/prepare_dataset.py \
  --source-dir "/content/drive/MyDrive/rice_disease_dataset/raw/Classification of Rice Diseases Using Leaf Images" \
  --data-root "/content/drive/MyDrive/rice_disease_dataset" \
  --seed 42
```

The script:
- discovers images under the three class folders;
- ignores non-image files and unrecognized folders;
- computes SHA-256 hashes and removes exact duplicate files within the same class before splitting;
- stops if identical image content has conflicting class labels;
- creates stratified 70% train, 15% validation, and 15% test manifests;
- writes `dataset_audit.json` with the observed class and split counts.

The split utility requires at least 14 unique images in each class so both validation and test subsets can retain all three classes under the two-stage stratified 70/15/15 split. For a robust experiment, verify that every class has enough independent images. Near-duplicate images, images from the same plant/session, and source leakage still require a separate visual/group-level audit; exact hashes alone cannot detect all such relationships.

## Install and run

```bash
pip install -e '.[dev]'
python scripts/train.py --config configs/default.yaml --model efficientnet_b0
python scripts/train.py --config configs/default.yaml --model vit_b16
python scripts/train.py --config configs/default.yaml --model hybrid
python scripts/evaluate.py --checkpoint PATH_TO_BEST_PT --config configs/default.yaml --split test
```

The three models use the same CSV split manifests. Validation macro-F1 selects checkpoints; the test split is reserved for final evaluation. Evaluation reports accuracy, macro precision/recall/F1, classification report, confusion matrix, and macro one-vs-rest ROC-AUC when all classes have sufficient positive and negative support.

## Research integrity

No raw dataset, synthetic sample, trained checkpoint, or experimental result is included in this repository. A code test is not a model experiment. Training and test metrics must only be reported after running the pipeline against the downloaded dataset and retaining the generated audit and split manifests.
