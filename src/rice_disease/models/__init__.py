from .efficientnet import build_efficientnet_b0
from .vit import build_vit_b16
from .hybrid import HybridCNNViT, build_hybrid


def build_model(name, num_classes=3, pretrained=True):
    key = name.lower()
    if key in {"efficientnet", "efficientnet_b0"}:
        return build_efficientnet_b0(num_classes, pretrained)
    if key in {"vit", "vit_b16"}:
        return build_vit_b16(num_classes, pretrained)
    if key in {"hybrid", "hybrid_cnn_vit"}:
        return build_hybrid(num_classes, pretrained)
    raise ValueError(f"Unknown model: {name}")
