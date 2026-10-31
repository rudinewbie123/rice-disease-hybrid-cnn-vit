"""Create leakage-aware stratified manifests from a Kaggle class-folder dataset.

The raw Kaggle download is never copied into this repository. Exact duplicate images
within the selected source are deduplicated before splitting; conflicting labels fail.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_ALIASES = {
    "blast": {"blast", "leaf blast", "rice blast"},
    "blight": {"blight", "leaf blight", "bacterial blight", "bacterial leaf blight"},
    "tungro": {"tungro", "rice tungro"},
}


def canonical_class(folder_name: str) -> str | None:
    normalized = " ".join(folder_name.lower().replace("_", " ").replace("-", " ").split())
    for label, aliases in CLASS_ALIASES.items():
        if normalized in aliases:
            return label
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_images(source_dir: Path, data_root: Path) -> tuple[list[dict], dict]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Dataset source directory does not exist: {source_dir}")
    records, hash_labels = [], {}
    ignored = 0
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = next((canonical_class(parent.name) for parent in path.parents
                      if parent == source_dir or source_dir in parent.parents
                      if canonical_class(parent.name) is not None), None)
        if label is None:
            ignored += 1
            continue
        digest = sha256_file(path)
        if digest in hash_labels:
            if hash_labels[digest] != label:
                raise ValueError(f"Identical image content has conflicting labels: {path}")
            continue
        hash_labels[digest] = label
        try:
            relative_path = path.resolve().relative_to(data_root.resolve()).as_posix()
        except ValueError as exc:
            raise ValueError("source_dir must be located inside data_root so manifests remain portable") from exc
        records.append({"image_path": relative_path, "label": label, "sha256": digest})
    counts = Counter(row["label"] for row in records)
    missing = set(CLASS_ALIASES) - set(counts)
    if missing:
        raise ValueError(f"Missing required class folders/images: {sorted(missing)}; found {dict(counts)}")
    if any(count < 3 for count in counts.values()):
        raise ValueError(f"Each class needs at least 3 unique images for 70/15/15 stratification: {dict(counts)}")
    return records, {"unique_image_counts": dict(sorted(counts.items())), "ignored_images": ignored}


def stratified_split(records: list[dict], seed: int = 42) -> dict[str, pd.DataFrame]:
    frame = pd.DataFrame(records)
    train, temporary = train_test_split(
        frame, test_size=0.30, random_state=seed, stratify=frame["label"]
    )
    validation, test = train_test_split(
        temporary, test_size=0.50, random_state=seed, stratify=temporary["label"]
    )
    result = {}
    for name, subset in (("train", train), ("val", validation), ("test", test)):
        result[name] = subset[["image_path", "label"]].sort_values(
            ["label", "image_path"]
        ).reset_index(drop=True)
    return result


def prepare_dataset(source_dir: Path, data_root: Path, split_dir: Path, seed: int = 42) -> dict:
    records, audit = discover_images(source_dir, data_root)
    splits = stratified_split(records, seed)
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_csv(split_dir / f"{name}.csv", index=False)
    audit["total_unique_images"] = len(records)
    audit["split_counts"] = {
        name: {"total": len(frame), "by_class": frame["label"].value_counts().sort_index().to_dict()}
        for name, frame in splits.items()
    }
    audit["seed"] = seed
    audit["ratios"] = {"train": 0.70, "val": 0.15, "test": 0.15}
    audit["deduplication"] = "Exact SHA-256 duplicates within same class retained once; conflicting labels rejected."
    (split_dir / "dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, help="Extracted Kaggle directory containing class folders")
    parser.add_argument("--data-root", required=True, help="Google Drive dataset root; source-dir must be inside it")
    parser.add_argument("--split-dir", default=None, help="Defaults to DATA_ROOT/splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(args.data_root).expanduser().resolve()
    source = Path(args.source_dir).expanduser().resolve()
    output = Path(args.split_dir).expanduser().resolve() if args.split_dir else root / "splits"
    audit = prepare_dataset(source, root, output, args.seed)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
