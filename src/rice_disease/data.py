from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def build_transforms(image_size=224, train=False):
    ops = [transforms.Resize((image_size, image_size))]
    if train:
        ops += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(12),
                transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.10)]
    ops += [transforms.ToTensor(), transforms.Normalize((.485,.456,.406),(.229,.224,.225))]
    return transforms.Compose(ops)


class RiceLeafDataset(Dataset):
    def __init__(self, manifest, data_root, class_names, transform=None):
        self.manifest_path, self.root = Path(manifest), Path(data_root)
        self.names = list(class_names)
        self.label_to_index = {name: i for i, name in enumerate(self.names)}
        self.transform = transform
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Split manifest not found: {self.manifest_path}")
        self.frame = pd.read_csv(self.manifest_path)
        required = {"image_path", "label"}
        if not required.issubset(self.frame.columns):
            raise ValueError(f"Manifest must contain columns: {sorted(required)}")
        if self.frame.empty:
            raise ValueError(f"Empty split manifest: {self.manifest_path}")
        unknown = set(self.frame.label.astype(str)) - set(self.label_to_index)
        if unknown:
            raise ValueError(f"Unknown labels: {sorted(unknown)}")

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = Path(str(row.image_path))
        if not path.is_absolute():
            path = self.root / path
        if not path.is_file():
            raise FileNotFoundError(f"Image referenced by manifest not found: {path}")
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(self.label_to_index[str(row.label)], dtype=torch.long)


def make_datasets(config):
    root = Path(config["data_root"])
    split_dir = root / config.get("split_dir", "splits")
    names = config["class_names"]
    return {split: RiceLeafDataset(split_dir / config[f"{split}_csv"], root, names,
            build_transforms(config.get("image_size",224), train=(split == "train")))
            for split in ("train", "val", "test")}
