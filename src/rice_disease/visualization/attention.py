import torch


def hybrid_attention_rollout(model,image):
    """Roll out actual per-head self-attention matrices from each hybrid transformer block."""
    model.eval()
    with torch.no_grad():
        maps=model.get_attention_maps(image)
        if not maps: raise RuntimeError("No attention maps returned")
        batch,_,tokens,_=maps[0].shape
        identity=torch.eye(tokens,device=image.device).unsqueeze(0).expand(batch,-1,-1)
        rollout=identity
        for attention in maps:
            averaged=attention.mean(1)+torch.eye(tokens,device=image.device).unsqueeze(0)
            averaged=averaged/averaged.sum(-1,keepdim=True)
            rollout=torch.bmm(averaged,rollout)
        patches=rollout[:,0,1:]; side=int(patches.shape[-1]**.5)
        if side*side!=patches.shape[-1]: raise ValueError("Expected square patch grid")
        return patches.reshape(batch,1,side,side)
