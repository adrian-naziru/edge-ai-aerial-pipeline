import os
import random
import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from contextlib import nullcontext
from tqdm import tqdm

from dataset_torch import make_dataloader
from mask2former_flexible import Mask2Former
from losses_full import HybridLoss


# CONFIG
BACKBONE = "tiny"      # "resnet18" sau "resnet50" / "tiny"
IMAGE_SIZE = 512           # 256 / 320 / 384 / 512
NUM_QUERIES = 25           # 16 / 25 / 50 / 100
TRANSFORMER_LAYERS = 3     # 2 / 3 / 6
BATCH_SIZE = 20
EPOCHS = 40
DATASET_FRACTION = 0.99


PALETTE_5 = np.array([
    [0,   0,   0],
    [255,   0,   0],
    [0,   255,   0],
    [0,     0, 255],
    [255, 255,   0],
], dtype=np.uint8)

# mIoU

def compute_mIoU(preds, targets, num_classes=5):
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()

    ious = []
    for c in range(num_classes):
        inter = np.logical_and(preds == c, targets == c).sum()
        union = np.logical_or(preds == c, targets == c).sum()
        if union > 0:
            ious.append(inter / union)

    return float(np.mean(ious)) if len(ious) else 0.0

# SAVE INFERENCE

@torch.no_grad()
def run_inference_and_save_single(model, device, img_path,
                                  save_folder="results_mask2former",
                                  epoch=None, idx=None):

    os.makedirs(save_folder, exist_ok=True)

    orig = Image.open(img_path).convert("RGB")
    img_np = np.array(orig) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = (img_np - mean) / std
    img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)

    logits = model(img_t)
    logits = F.interpolate(logits, size=orig.size[::-1], mode="bilinear")

    preds = torch.argmax(logits, dim=1).cpu().numpy()[0]
    overlay = cv2.addWeighted(np.array(orig), 0.6, PALETTE_5[preds], 0.4, 0)

    if epoch is None:
        s1 = f"{save_folder}/seg.png"
        o1 = f"{save_folder}/orig.png"
    else:
        s1 = f"{save_folder}/seg_epoch_{epoch:02d}_{idx}.png"
        o1 = f"{save_folder}/orig_epoch_{epoch:02d}_{idx}.png"

    Image.fromarray(np.array(orig)).save(o1)
    Image.fromarray(overlay).save(s1)


# TRAIN

def train_one_epoch(model, loader, optimizer, scaler, criterion, device):
    model.train()
    total = 0.0

    for imgs, masks in tqdm(loader, desc="Training", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            with autocast("cuda"):
                logits = model(imgs)
                logits = F.interpolate(
                    logits, size=masks.shape[-2:], mode="bilinear"
                )

            loss = criterion(logits.float(), masks)

            if not torch.isfinite(loss):
                print(" Non-finite loss, skipping batch:", loss.item())
                continue

            scaler.scale(loss).backward()

            # unscale + gradient clipping
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

        else:
            logits = model(imgs)
            logits = F.interpolate(
                logits, size=masks.shape[-2:], mode="bilinear"
            )
            loss = criterion(logits, masks)

            if not torch.isfinite(loss):
                print("Non-finite loss on CPU, skipping batch:", loss.item())
                continue

            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total += loss.item()

    return total / len(loader)

# VALIDATION

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    miou_sum = 0.0
    n = 0

    for imgs, masks in tqdm(loader, desc="Validation", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)

        logits = model(imgs)
        logits = F.interpolate(
            logits, size=masks.shape[-2:], mode="bilinear"
        )

        loss = criterion(logits.float(), masks)
        preds = torch.argmax(logits, dim=1)
        miou = compute_mIoU(preds, masks)

        loss_sum += loss.item()
        miou_sum += miou
        n += 1

    return loss_sum / n, miou_sum / n

# MAIN

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)
    if device.type == "cuda":
        print("Using CUDA device:", torch.cuda.get_device_name(0))
        print("Is CUDA active NOW:", torch.cuda.is_initialized())

    # LOAD TRAIN LIST

    with open("dataset_landcover/output/train.txt") as f:
        full_list = [l.strip() for l in f if l.strip()]

    random.shuffle(full_list)
    N = int(len(full_list) * DATASET_FRACTION)
    train_list = full_list[:N]

    print(f"[INFO] Using {N} samples from original {len(full_list)} "
          f"({DATASET_FRACTION*100:.0f}%)")

    train_loader = make_dataloader(train_list,
                                   batch_size=BATCH_SIZE,
                                   shuffle=True,
                                   img_size=IMAGE_SIZE,
                                   num_workers=4)

    val_loader = make_dataloader("dataset_landcover/output/val.txt",
                                 batch_size=BATCH_SIZE,
                                 shuffle=False,
                                 img_size=IMAGE_SIZE,
                                 num_workers=4)

    model = Mask2Former(
        num_classes=5,
        backbone_name=BACKBONE,
        num_queries=NUM_QUERIES,
        num_layers=TRANSFORMER_LAYERS
    ).to(device)
    print("MODEL RUNNING ON:", next(model.parameters()).device)

    class_weights = torch.tensor([1.0, 2.0, 1.5, 3.0, 2.5]).float().to(device)
    criterion = HybridLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    scaler = GradScaler() if device.type == "cuda" else None

    ckpt_path = "checkpoints_mask2former/last.pth"
    os.makedirs("checkpoints_mask2former", exist_ok=True)

    start_epoch = 1
    if os.path.exists(ckpt_path):
        print("[INFO] Loading checkpoint...")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        if scaler is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1

    # TRAIN LOOP

    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"\n[INFO] Epoch {epoch}/{EPOCHS}")

        tr = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device)
        va, miou = evaluate(model, val_loader, criterion, device)

        print(f"Train: {tr:.4f} | Val: {va:.4f} | mIoU: {miou:.4f}")


        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else {}
        }, ckpt_path)

        miou_str = f"{miou:.4f}"
        miou_clean = miou_str.replace(".", "_")

        epoch_ckpt_path = f"checkpoints_mask2former/model_epoch_{epoch:02d}_miou_{miou_clean}.pth"

        torch.save({
            "epoch": epoch,
            "miou": miou,
            "model": model.state_dict()
        }, epoch_ckpt_path)

        print(f"[INFO] Saved: {epoch_ckpt_path}")

        with open("dataset_landcover/output/val.txt") as f:
            val_names = [l.strip() for l in f if l.strip()]
        chosen = random.sample(val_names, 3)

        for i, name in enumerate(chosen):
            img = os.path.join("dataset_landcover/output/images", name)
            run_inference_and_save_single(model, device, img,
                                          epoch=epoch, idx=i)


if __name__ == "__main__":
    main()
