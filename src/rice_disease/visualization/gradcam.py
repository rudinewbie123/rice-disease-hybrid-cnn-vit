"""Grad-CAM for convolutional feature maps in CNN and hybrid models."""
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self.handle = target_layer.register_forward_hook(self._capture_activation)

    def _capture_activation(self, module, inputs, output):
        self.activations = output
        if output.requires_grad:
            output.register_hook(self._capture_gradient)

    def _capture_gradient(self, gradient):
        self.gradients = gradient

    def __call__(self, image, class_index=None):
        self.model.zero_grad(set_to_none=True)
        self.activations = self.gradients = None
        logits = self.model(image)
        if class_index is None:
            class_index = int(logits.argmax(dim=1)[0])
        if not 0 <= int(class_index) < logits.shape[1]:
            raise ValueError("class_index is outside the model output range.")
        logits[:, int(class_index)].sum().backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True).relu()
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear",
                             align_corners=False)[0, 0].detach().cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    def close(self):
        self.handle.remove()


def tensor_to_rgb(image_tensor):
    """Undo ImageNet normalization and return an HWC uint8 RGB image."""
    mean = image_tensor.new_tensor([.485, .456, .406])[:, None, None]
    std = image_tensor.new_tensor([.229, .224, .225])[:, None, None]
    image = (image_tensor.detach().cpu() * std.cpu() + mean.cpu()).clamp(0, 1)
    return (image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def overlay_heatmap(image_rgb, heatmap, alpha=.4):
    import matplotlib.pyplot as plt
    heat = plt.get_cmap("jet")(heatmap)[..., :3]
    return np.clip((1-alpha) * (image_rgb.astype(np.float32) / 255.) + alpha * heat,
                   0, 1)
