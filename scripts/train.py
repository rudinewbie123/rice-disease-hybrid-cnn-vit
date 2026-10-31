import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from rice_disease.config import load_config
from rice_disease.data import make_datasets
from rice_disease.engine import fit,set_seed
from rice_disease.models import build_model


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/default.yaml")
    parser.add_argument("--model",choices=["efficientnet_b0","vit_b16","hybrid"])
    args=parser.parse_args(); config=load_config(args.config)
    if args.model: config["model"]=args.model
    set_seed(int(config.get("seed",42))); datasets=make_datasets(config)
    loaders={name:DataLoader(ds,batch_size=int(config["batch_size"]),shuffle=name=="train",
        num_workers=int(config.get("num_workers",2)),pin_memory=torch.cuda.is_available())
        for name,ds in datasets.items()}
    model=build_model(config["model"],int(config["num_classes"]),bool(config.get("pretrained",True)))
    fit(model,loaders["train"],loaders["val"],config,Path(config["output_dir"])/config["model"])

if __name__=="__main__": main()
