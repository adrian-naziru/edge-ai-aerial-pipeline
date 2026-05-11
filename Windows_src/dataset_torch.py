import os
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset, DataLoader

IMG_DIR = "dataset_landcover/output/images"
MASK_DIR = "dataset_landcover/output/masks"


class SegmentationDataset(Dataset):
    def __init__(self, names, img_dir=IMG_DIR, mask_dir=MASK_DIR, img_size=512):

        self.names = names
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]

        img_path = os.path.join(self.img_dir, name)
        mask_path = os.path.join(self.mask_dir, name)

        if not img_path.endswith(".png"):
            img_path += ".png"
        if not mask_path.endswith(".png"):
            mask_path += ".png"

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise FileNotFoundError(img_path)
        if mask is None:
            raise FileNotFoundError(mask_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size),
                             interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.img_size, self.img_size),
                              interpolation=cv2.INTER_NEAREST)

        img = img.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        img = (img - mean) / std
        mask = mask.astype(np.int64)

        # [H,W,C] → [C,H,W]
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        mask = torch.from_numpy(mask).contiguous()

        return img, mask


def make_dataloader(list_or_path,
                    batch_size=4,
                    shuffle=True,
                    img_size=512,
                    num_workers=4):

    # 1. Path txt
    if isinstance(list_or_path, str) and os.path.isfile(list_or_path):
        with open(list_or_path, "r") as f:
            names = [l.strip() for l in f if l.strip()]

    elif isinstance(list_or_path, list):
        names = list_or_path

    else:
        raise ValueError("make_dataloader: need list or .txt")

    dataset = SegmentationDataset(names, img_size=img_size)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    return loader
