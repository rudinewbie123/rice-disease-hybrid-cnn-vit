"""Prepare Kaggle rice disease image folders into stratified train/val/test manifests."""
import argparse
import json
from pathlib import Path

from rice_disease.dataset_prep import prepare_dataset


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
    print(json.dumps(prepare_dataset(source, root, output, args.seed), indent=2))


if __name__ == "__main__":
    main()
