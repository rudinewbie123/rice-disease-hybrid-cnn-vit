from torch import nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


def build_vit_b16(num_classes=3, pretrained=True):
    weights = ViT_B_16_Weights.DEFAULT if pretrained else None
    model = vit_b_16(weights=weights)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model
