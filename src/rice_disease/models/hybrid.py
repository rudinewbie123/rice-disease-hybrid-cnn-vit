import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim*4), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(embed_dim*4, embed_dim), nn.Dropout(dropout))

    def forward(self, x, return_attention=False):
        z = self.norm1(x)
        attended, weights = self.attention(z, z, z, need_weights=return_attention,
                                            average_attn_weights=False)
        x = x + attended
        return x + self.mlp(self.norm2(x)), weights


class HybridCNNViT(nn.Module):
    """EfficientNet-B0 local encoder plus transformer global-context encoder."""
    def __init__(self, num_classes=3, pretrained=True, embed_dim=256, num_heads=8,
                 depth=4, dropout=.1):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.cnn_features = efficientnet_b0(weights=weights).features
        self.projection = nn.Conv2d(1280, embed_dim, 1)
        self.cls_token = nn.Parameter(torch.zeros(1,1,embed_dim))
        self.position = nn.Parameter(torch.zeros(1,50,embed_dim))
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim,num_heads,dropout) for _ in range(depth)])
        self.norm, self.dropout = nn.LayerNorm(embed_dim), nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim,num_classes)
        nn.init.trunc_normal_(self.cls_token,std=.02)
        nn.init.trunc_normal_(self.position,std=.02)

    def _position_for(self, count):
        if count == self.position.shape[1]:
            return self.position
        spatial = self.position[:,1:].transpose(1,2)
        side = int(spatial.shape[-1]**.5)
        spatial = spatial.reshape(1,-1,side,side)
        target = int((count-1)**.5)
        if target*target != count-1:
            raise ValueError("Expected square patch grid")
        spatial = torch.nn.functional.interpolate(spatial,size=(target,target),mode="bilinear",align_corners=False)
        return torch.cat((self.position[:,:1],spatial.flatten(2).transpose(1,2)),dim=1)

    def forward_features(self, x, return_attention=False):
        x = self.projection(self.cnn_features(x)).flatten(2).transpose(1,2)
        x = torch.cat((self.cls_token.expand(x.shape[0],-1,-1),x),dim=1)
        x = x + self._position_for(x.shape[1])
        maps=[]
        for block in self.blocks:
            x, weights = block(x,return_attention)
            if return_attention: maps.append(weights)
        features = self.norm(x)[:,0]
        return (features,maps) if return_attention else features

    def forward(self,x):
        return self.classifier(self.dropout(self.forward_features(x)))

    def get_attention_maps(self,x):
        return self.forward_features(x,return_attention=True)[1]


def build_hybrid(num_classes=3,pretrained=True,**kwargs):
    return HybridCNNViT(num_classes,pretrained,**kwargs)
