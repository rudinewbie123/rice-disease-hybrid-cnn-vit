"""Evaluate a saved model checkpoint on a fixed validation or test split."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rice_disease.config import load_config
from rice_disease.data import RiceLeafDataset, build_transforms
from rice_disease.metrics import calculate_metrics, save_confusion_matrix, save_roc_curves
from rice_disease.models import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    names = checkpoint.get("class_names", config["class_names"])
    root = Path(config["data_root"])
    manifest = root / config.get("split_dir", "splits") / config[f"{args.split}_csv"]
    dataset = RiceLeafDataset(manifest, root, names,
                              build_transforms(config["image_size"], False))
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False,
                        num_workers=int(config.get("num_workers", 2)))
    model = build_model(checkpoint["model_name"], len(names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    truth, probabilities = [], []
    with torch.inference_mode():
        for images, labels in loader:
            probabilities.extend(torch.softmax(model(images.to(device)), dim=1).cpu().tolist())
            truth.extend(labels.tolist())
    result = calculate_metrics(truth, probabilities, names)
    output = Path(config["output_dir"]) / f"{checkpoint['model_name']}_{args.split}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "classification_report.json").write_text(
        json.dumps(result["classification_report"], indent=2), encoding="utf-8")
    save_confusion_matrix(result["confusion_matrix"], names, output / "confusion_matrix.png")
    save_roc_curves(truth, probabilities, names, output / "roc_curves_ovr.png")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "classification_report"}, indent=2))


if __name__ == "__main__":
    main()
