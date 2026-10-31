import pytest
import torch
from rice_disease.models import build_model
from rice_disease.models.hybrid import build_hybrid
from rice_disease.visualization.attention import hybrid_attention_rollout


@pytest.mark.parametrize("name",["efficientnet_b0","vit_b16","hybrid"])
def test_model_output_shape(name):
    model=build_model(name,3,False).eval()
    with torch.no_grad(): output=model(torch.randn(1,3,224,224))
    assert output.shape==(1,3)


def test_hybrid_attention_rollout_shape():
    model=build_hybrid(3,False,embed_dim=64,num_heads=4,depth=2).eval(); image=torch.randn(1,3,224,224)
    with torch.no_grad(): maps=model.get_attention_maps(image); rollout=hybrid_attention_rollout(model,image)
    assert len(maps)==2 and maps[0].shape==(1,4,50,50)
    assert rollout.shape==(1,1,7,7)
