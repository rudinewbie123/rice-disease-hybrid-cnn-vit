"""Attention rollout for torchvision ViT-B/16 and the custom Hybrid CNN-ViT."""
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn import functional as nnf


def hybrid_attention_rollout(model, image):
    """Roll out actual per-head self-attention matrices from hybrid transformer blocks."""
    model.eval()
    with torch.no_grad():
        maps = model.get_attention_maps(image)
        if not maps:
            raise RuntimeError("No attention maps returned")
        batch, _, tokens, _ = maps[0].shape
        identity = torch.eye(tokens, device=image.device).unsqueeze(0).expand(batch, -1, -1)
        rollout = identity
        for attention in maps:
            averaged = attention.mean(1) + torch.eye(tokens, device=image.device).unsqueeze(0)
            averaged = averaged / averaged.sum(-1, keepdim=True)
            rollout = torch.bmm(averaged, rollout)
        patches = rollout[:, 0, 1:]
        side = int(patches.shape[-1] ** 0.5)
        if side * side != patches.shape[-1]:
            raise ValueError("Expected square patch grid")
        return patches.reshape(batch, 1, side, side)


def vit_attention_rollout(model, image):
    """Extract per-head QK attention from torchvision ViT encoder blocks and roll it out.

    Torchvision's stock EncoderBlock calls MultiheadAttention with need_weights=False.
    This function reconstructs attention probabilities from the module's actual Q/K
    projections using a forward-pre-hook, without modifying model weights or outputs.
    """
    model.eval()
    layers = getattr(getattr(model, "encoder", None), "layers", None)
    if layers is None:
        raise TypeError("Expected a torchvision VisionTransformer with encoder.layers.")
    captured = []

    def capture_attention(module, args):
        query = args[0]  # torchvision ViT uses sequence-first: [tokens, batch, embedding]
        if query.ndim != 3:
            raise ValueError("Unexpected ViT attention input shape.")
        embed_dim = query.shape[-1]
        heads = module.num_heads
        head_dim = embed_dim // heads
        if getattr(module, "in_proj_weight", None) is None:
            raise NotImplementedError("ViT attention requires packed QKV projection weights.")
        qkv = nnf.linear(query, module.in_proj_weight, module.in_proj_bias)
        q, k, _ = qkv.chunk(3, dim=-1)
        batch = query.shape[1]
        q = q.reshape(query.shape[0], batch, heads, head_dim).permute(1, 2, 0, 3)
        k = k.reshape(query.shape[0], batch, heads, head_dim).permute(1, 2, 0, 3)
        scores = torch.matmul(q / (head_dim ** 0.5), k.transpose(-2, -1))
        captured.append(scores.softmax(dim=-1).detach())

    handles = []
    try:
        for block in layers:
            handles.append(block.self_attention.register_forward_pre_hook(capture_attention))
        with torch.no_grad():
            model(image)
    finally:
        for handle in handles:
            handle.remove()
    if not captured:
        raise RuntimeError("No attention matrices were captured from ViT.")
    batch, _, tokens, _ = captured[0].shape
    identity = torch.eye(tokens, device=image.device).unsqueeze(0).expand(batch, -1, -1)
    rollout = identity
    for attention in captured:
        averaged = attention.mean(dim=1) + identity
        averaged = averaged / averaged.sum(dim=-1, keepdim=True)
        rollout = torch.bmm(averaged, rollout)
    patches = rollout[:, 0, 1:]
    side = int(patches.shape[-1] ** 0.5)
    if side * side != patches.shape[-1]:
        raise ValueError("ViT patch tokens do not form a square grid.")
    return patches.reshape(batch, 1, side, side)


def save_attention_visualization(model, image, output_path, original_rgb=None,
                                  method="hybrid"):
    import matplotlib.pyplot as plt
    rollout = vit_attention_rollout if method == "vit" else hybrid_attention_rollout
    attention = rollout(model, image)
    attention = F.interpolate(attention, size=image.shape[-2:], mode="bilinear",
                               align_corners=False)[0, 0].cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    if original_rgb is None:
        ax.imshow(attention, cmap="inferno")
    else:
        ax.imshow(original_rgb)
        ax.imshow(attention, cmap="inferno", alpha=.45)
    ax.set_title(f"{method.upper()} Attention Rollout")
    ax.axis("off")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path
