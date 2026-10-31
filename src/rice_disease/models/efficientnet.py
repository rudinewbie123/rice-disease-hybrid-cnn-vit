from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def build_efficientnet_b0(num_classes=3, pretrained=True):
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model
