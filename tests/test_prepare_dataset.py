from pathlib import Path

import pandas as pd
from PIL import Image

from rice_disease.prepare_dataset import prepare_dataset


def _make_image(path: Path, value: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), (value, value, value)).save(path)


def test_stratified_prepare_creates_three_splits_and_audit(tmp_path):
    root = tmp_path / "drive_data"
    source = root / "raw" / "rice"
    for class_index, label in enumerate(("Blast", "Blight", "Tungro")):
        for i in range(20):
            _make_image(source / label / f"{i}.png", class_index * 60 + i)
    audit = prepare_dataset(source, root, root / "splits", seed=42)
    train = pd.read_csv(root / "splits" / "train.csv")
    val = pd.read_csv(root / "splits" / "val.csv")
    test = pd.read_csv(root / "splits" / "test.csv")
    assert set(train.label) == {"blast", "blight", "tungro"}
    assert len(train) + len(val) + len(test) == 60
    assert set(train.image_path).isdisjoint(set(val.image_path))
    assert set(train.image_path).isdisjoint(set(test.image_path))
    assert set(val.image_path).isdisjoint(set(test.image_path))
    for frame in (train, val, test):
        assert set(frame.label.value_counts().index) == {"blast", "blight", "tungro"}
    assert audit["total_unique_images"] == 60


def test_exact_duplicate_is_removed_before_split(tmp_path):
    root = tmp_path / "drive_data"
    source = root / "raw" / "rice"
    for class_index, label in enumerate(("Blast", "Blight", "Tungro")):
        for i in range(4):
            _make_image(source / label / f"{i}.png", class_index * 70 + i)
    duplicate = source / "Blast" / "copy.png"
    duplicate.parent.mkdir(parents=True, exist_ok=True)
    (source / "Blast" / "0.png").replace(duplicate)
    (source / "Blast" / "copy.png").replace(source / "Blast" / "0.png")
    audit = prepare_dataset(source, root, root / "splits", seed=7)
    assert audit["unique_image_counts"]["blast"] == 4
    assert audit["total_unique_images"] == 12
