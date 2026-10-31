import torch
import torch.nn.functional as F


def hybrid_attention_rollout(model, image):
    """Roll out actual per-head self-attention matrices from each hybrid transformer block."""
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


def save_attention_visualization(model, image, output_path, original_rgb=None):
    """Save a resized attention-rollout heatmap, optionally overlaid on the source RGB image."""
    import matplotlib.pyplot as plt

    attention = hybrid_attention_rollout(model, image)
    attention = F.interpolate(attention, size=image.shape[-2:], mode="bilinear",
                              align_corners=False)[0, 0].cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    if original_rgb is None:
        ax.imshow(attention, cmap="inferno")
    else:
        ax.imshow(original_rgb)
        ax.imshow(attention, cmap="inferno", alpha=0.45)
    ax.set_title("Hybrid CNN–ViT Attention Rollout")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path
