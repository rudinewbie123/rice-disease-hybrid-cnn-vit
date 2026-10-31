import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self,model,target_layer):
        self.model=model; self.activations=None; self.gradients=None
        self.handles=[target_layer.register_forward_hook(self._activation),
                      target_layer.register_full_backward_hook(self._gradient)]
    def _activation(self,module,inputs,output): self.activations=output
    def _gradient(self,module,grad_input,grad_output): self.gradients=grad_output[0]
    def __call__(self,image,class_index=None):
        self.model.zero_grad(set_to_none=True); logits=self.model(image)
        if class_index is None: class_index=int(logits.argmax(1)[0])
        logits[:,class_index].sum().backward()
        weights=self.gradients.mean((2,3),keepdim=True)
        cam=(weights*self.activations).sum(1,keepdim=True).relu()
        cam=F.interpolate(cam,size=image.shape[-2:],mode="bilinear",align_corners=False)[0,0].detach().cpu().numpy()
        return (cam-cam.min())/(cam.max()-cam.min()+1e-8)
    def close(self):
        for handle in self.handles: handle.remove()


def overlay_heatmap(image_rgb,heatmap,alpha=.4):
    import matplotlib.pyplot as plt
    heat=plt.get_cmap("jet")(heatmap)[...,:3]
    return np.clip((1-alpha)*(image_rgb.astype(np.float32)/255.)+alpha*heat,0,1)
