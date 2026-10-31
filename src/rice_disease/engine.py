import random
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from tqdm import tqdm


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def run_epoch(model,loader,criterion,device,optimizer=None):
    training = optimizer is not None
    model.train(training); total=0.; truth=[]; pred=[]
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images,labels in tqdm(loader,leave=False):
            images,labels=images.to(device),labels.to(device)
            if training: optimizer.zero_grad(set_to_none=True)
            logits=model(images); loss=criterion(logits,labels)
            if training: loss.backward(); optimizer.step()
            total += loss.item()*labels.size(0)
            truth.extend(labels.cpu().tolist()); pred.extend(logits.argmax(1).detach().cpu().tolist())
    return {"loss":total/len(loader.dataset),"macro_f1":f1_score(truth,pred,average="macro",zero_division=0)}


def fit(model,train_loader,val_loader,config,output_dir):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.get("learning_rate",1e-4),
                                weight_decay=config.get("weight_decay",1e-4))
    criterion=nn.CrossEntropyLoss(); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    best=-1.; history=[]
    for epoch in range(1,int(config["epochs"])+1):
        tr=run_epoch(model,train_loader,criterion,device,optimizer)
        va=run_epoch(model,val_loader,criterion,device)
        row={"epoch":epoch,"train_loss":tr["loss"],"train_macro_f1":tr["macro_f1"],
             "val_loss":va["loss"],"val_macro_f1":va["macro_f1"]}
        history.append(row); print(row)
        if va["macro_f1"]>best:
            best=va["macro_f1"]
            torch.save({"model_state_dict":model.state_dict(),"model_name":config["model"],
                        "class_names":config["class_names"],"config":config,"best_val_macro_f1":best},out/"best.pt")
    import json
    (out/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
    return history
