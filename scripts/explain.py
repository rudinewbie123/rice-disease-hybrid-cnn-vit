"""Generate Grad-CAM and transformer attention maps for held-out images."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from rice_disease.config import load_config
from rice_disease.data import RiceLeafDataset, build_transforms
from rice_disease.models import build_model
from rice_disease.visualization.attention import save_attention_visualization
from rice_disease.visualization.gradcam import GradCAM, overlay_heatmap, tensor_to_rgb


def target_layer_for(model, model_name):
    if model_name in {"efficientnet_b0", "efficientnet"}:
        return model.features[-1]
    if model_name in {"hybrid", "hybrid_cnn_vit"}:
        return model.cnn_features[-1]
    raise ValueError(f"Grad-CAM is not configured for model: {model_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--max-samples", type=int, default=12)
    parser.add_argument("--method", choices=["all", "gradcam", "attention"], default="all")
    args = parser.parse_args()
    if args.max_samples < 1:
        raise ValueError("--max-samples must be positive.")
    config = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_name = checkpoint["model_name"]
    names = checkpoint.get("class_names", config["class_names"])
    root = Path(config["data_root"])
    manifest = root / config.get("split_dir", "splits") / config[f"{args.split}_csv"]
    dataset = RiceLeafDataset(manifest, root, names, build_transforms(config["image_size"], False))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = build_model(model_name, len(names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    output = Path(config["output_dir"]) / f"{model_name}_{args.split}" / "explainability"
    output.mkdir(parents=True, exist_ok=True)
    is_vit = model_name in {"vit", "vit_b16"}
    is_hybrid = model_name in {"hybrid", "hybrid_cnn_vit"}
    gradcam = None
    if args.method in {"all", "gradcam"} and not is_vit:
        gradcam = GradCAM(model, target_layer_for(model, model_name))
    if args.method == "gradcam" and is_vit:
        raise ValueError("ViT has no convolutional feature map; use --method attention.")
    count = 0
    try:
        for index, (images, labels) in enumerate(loader):
            if count >= args.max_samples:
                break
            image = images.to(device)
            rgb = tensor_to_rgb(images[0])
            with torch.no_grad():
                logits = model(image)
                predicted = int(logits.argmax(dim=1)[0])
            actual = int(labels[0])
            stem = f"sample_{index:04d}_true-{names[actual]}_pred-{names[predicted]}"
            if gradcam is not None:
                heatmap = gradcam(image, predicted)
                overlay = overlay_heatmap(rgb, heatmap)
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(overlay)
                ax.set_title(f"Grad-CAM | true={names[actual]} | pred={names[predicted]}")
                ax.axis("off")
                fig.tight_layout()
                fig.savefig(output / f"{stem}_gradcam.png", dpi=180, bbox_inches="tight")
                plt.close(fig)
            if args.method in {"all", "attention"} and (is_vit or is_hybrid):
                save_attention_visualization(
                    model, image, output / f"{stem}_attention.png", rgb,
                    method="vit" if is_vit else "hybrid",
                )
            elif args.method == "attention" and not (is_vit or is_hybrid):
                raise ValueError("EfficientNet-B0 has no transformer attention; use --method gradcam.")
            count += 1
    finally:
        if gradcam is not None:
            gradcam.close()
    print(f"Saved explainability outputs for {count} real {args.split} samples to {output}")


if __name__ == "__main__":
    main()
