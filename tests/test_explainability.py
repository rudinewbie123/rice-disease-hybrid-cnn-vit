import pytest
import torch
from torchvision.models import vit_b_16

from rice_disease.models.hybrid import build_hybrid
from rice_disease.visualization.attention import hybrid_attention_rollout, vit_attention_rollout
from rice_disease.visualization.gradcam import GradCAM


def test_hybrid_attention_rollout_shape():
    model = build_hybrid(3, False, embed_dim=64, num_heads=4, depth=2).eval()
    image = torch.randn(1,3,224,224)
    with torch.no_grad():
        maps = model.get_attention_maps(image)
        rollout = hybrid_attention_rollout(model, image)
    assert len(maps) == 2 and maps[0].shape == (1,4,50,50)
    assert rollout.shape == (1,1,7,7)


def test_vit_attention_rollout_shape():
    model = vit_b_16(weights=None).eval()
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, 3)
    image = torch.randn(1,3,224,224)
    rollout = vit_attention_rollout(model, image)
    assert rollout.shape == (1,1,14,14)
    assert torch.isfinite(rollout).all()


@pytest.mark.parametrize("kind", ["efficientnet", "hybrid"])
def test_gradcam_produces_finite_heatmap(kind):
    from torchvision.models import efficientnet_b0
    if kind == "efficientnet":
        model = efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 3)
        layer = model.features[-1]
    else:
        model = build_hybrid(3, False, embed_dim=64, num_heads=4, depth=1)
        layer = model.cnn_features[-1]
    model.eval()
    cam = GradCAM(model, layer)
    try:
        heatmap = cam(torch.randn(1,3,224,224))
        assert heatmap.shape == (224,224)
        assert torch.isfinite(torch.from_numpy(heatmap)).all()
    finally:
        cam.close()
