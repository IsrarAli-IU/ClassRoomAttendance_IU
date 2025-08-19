#!/usr/bin/env python3
"""
ArcFace (ResNet100) TRAINING — auto-configured for your folder layout

This script assumes the following **local** layout (no CLI args needed):

project_root/
├── train_arcface_resnet100.py   ← this file
├── labels.csv                   ← in the same folder as the script
└── dataset/                     ← folder that contains all student subfolders/images
    ├── 63018_Anisha_Sarhadi/
    ├── 61412_Areka_Raza_Hashmi/
    └── ...

It will:
  1) Read `labels.csv` and auto-resolve image paths whether they are like
     `63018_xxx/img1.jpg` **or** `dataset/63018_xxx/img1.jpg`.
  2) Detect + align faces to 112×112 using **RetinaFace** (insightface FaceAnalysis).
  3) Fine‑tune **ArcFace** (iResNet‑100 backbone + ArcMargin head) on your students.
  4) Save artifacts to `./arcface_artifacts/` by default:
       - ckpt_best.pth
       - prototypes.json (mean 512‑D embedding per student)
       - embeddings.csv   (per‑image embeddings for quick audits)

Dependencies (install on your GPU machine):
    pip install "torch==2.3.*" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install insightface opencv-python pandas numpy scikit-learn tqdm albumentations timm

Notes:
- First run will download model packs (RetinaFace + iResNet100) to ~/.insightface/models .
- For small class rosters, start with **frozen backbone** (only ArcFace head trains). You can unfreeze later.
- If your CPU-only environment is used, it will run but much slower; use a CUDA GPU if possible.
"""
import os
import math
import json
import time
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ---- InsightFace (detector + pretrained backbone) ----
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# =====================
# CONFIG (edit here)
# =====================
ROOT_DIR       = Path(__file__).parent.resolve()
LABELS_CSV     = ROOT_DIR / "labels.csv"
IMAGES_ROOT    = ROOT_DIR / "dataset"       # folder containing all student subfolders/images
OUT_DIR        = ROOT_DIR / "arcface_artifacts"
EPOCHS         = 10
BATCH_SIZE     = 32
LR             = 1e-3
ARC_S          = 64.0      # ArcFace scale
ARC_M          = 0.5       # ArcFace margin
FREEZE_BACKBONE = True     # start with head-only fine‑tuning
NUM_WORKERS    = 4         # reduce to 0 on Windows if you see DataLoader issues
SEED           = 42

# =====================
# Utils
# =====================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# =====================
# ArcMargin (ArcFace head)
# =====================
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features: int, out_features: int, s: float = 64.0, m: float = 0.5, easy_margin: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input: torch.Tensor, label: torch.Tensor):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(torch.clamp(1.0 - cosine ** 2, min=1e-9))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

# =====================
# Dataset
# =====================
class FacesFromCSV(Dataset):
    def __init__(self, labels_csv: Path, images_root: Path, face_app: FaceAnalysis, out_size: int = 112, augment: bool = True):
        self.df = pd.read_csv(labels_csv)
        self.images_root = Path(images_root)
        self.face_app = face_app
        self.out_size = out_size
        self.augment = augment

        # Map student_id → class index 0..C-1
        sids = sorted(self.df["student_id"].unique().tolist())
        self.sid2idx = {int(s): i for i, s in enumerate(sids)}

        # Build a resolved file list up-front (robust to either path style)
        self.items = []  # list of (abs_path, class_idx, student_id)
        missing = 0
        for _, row in self.df.iterrows():
            sid = int(row['student_id'])
            cls_idx = self.sid2idx[sid]
            rel = str(row['image_path']).lstrip("./")
            # support two styles: "dataset/..." or just "folder/file.jpg"
            p1 = (self.images_root.parent / rel) if rel.startswith("dataset/") else (self.images_root / rel)
            if not p1.exists():
                # try inside student folder if only folder name was given
                p_try = self.images_root / rel
                if p_try.exists():
                    p1 = p_try
            if p1.exists():
                self.items.append((p1, cls_idx, sid))
            else:
                missing += 1
        if missing:
            print(f"[WARN] {missing} images referenced in CSV could not be found. They will be skipped.")
        if not self.items:
            raise RuntimeError("No images found. Check your labels.csv and dataset/ structure.")

        # Light augmentations
        aug = []
        if augment:
            aug += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.15, contrast=0.15)
            ]
        self.aug_tf = transforms.Compose(aug) if aug else None
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # → [-1,1]
        ])

    def __len__(self):
        return len(self.items)

    def _align_face(self, img_bgr: np.ndarray) -> np.ndarray:
        preds = self.face_app.get(img_bgr)
        if len(preds) == 0:
            # fallback: center-crop
            h, w = img_bgr.shape[:2]
            m = min(h, w)
            y0 = (h - m) // 2
            x0 = (w - m) // 2
            crop = img_bgr[y0:y0 + m, x0:x0 + m]
            return cv2.resize(crop, (self.out_size, self.out_size), interpolation=cv2.INTER_AREA)
        preds.sort(key=lambda d: (d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]), reverse=True)
        kps = preds[0]['kps']
        src = np.array(kps, dtype=np.float32)
        dst = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ], dtype=np.float32)
        M = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)[0]
        aligned = cv2.warpAffine(img_bgr, M, (self.out_size, self.out_size))
        return aligned

    def __getitem__(self, idx: int):
        path, cls_idx, sid = self.items[idx]
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read {path}")
        aligned = self._align_face(img_bgr)
        img = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        if self.aug_tf:
            img = self.aug_tf(transforms.ToPILImage()(img))
            img = np.array(img)
        x = self.to_tensor(img)
        y = cls_idx
        return x, y, sid

# =====================
# Model wrapper
# =====================
class ArcFaceModel(nn.Module):
    def __init__(self, num_classes: int, feat_dim: int = 512, s: float = ARC_S, m: float = ARC_M):
        super().__init__()
        # Pretrained iResNet100
        self.backbone = get_model('glint360k_r100_fc')  # downloads on first use
        self.backbone.eval()
        self.head = ArcMarginProduct(in_features=feat_dim, out_features=num_classes, s=s, m=m)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None):
        feat = self.backbone.forward(x)  # [B,512]
        feat = F.normalize(feat)
        if labels is None:
            return feat
        logits = self.head(feat, labels)
        return logits, feat

# =====================
# Train / Eval helpers
# =====================
@torch.no_grad()
def compute_prototypes(backbone: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    backbone.eval()
    sums, counts = {}, {}
    for x, y, sid in tqdm(loader, desc="Prototypes", leave=False):
        x = x.to(device)
        f = F.normalize(backbone.forward(x))
        f = f.detach().cpu().numpy()
        for i in range(f.shape[0]):
            key = int(sid[i])
            sums[key] = sums.get(key, 0) + f[i]
            counts[key] = counts.get(key, 0) + 1
    return {k: (sums[k]/counts[k]).tolist() for k in sums}


def main():
    set_seed(SEED)
    ensure_dir(OUT_DIR)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Face detector/alignment
    face_app = FaceAnalysis(name='buffalo_l')
    face_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640,640))

    # Dataset & split (80/20 per student by row order)
    ds = FacesFromCSV(LABELS_CSV, IMAGES_ROOT, face_app, out_size=112, augment=True)
    num_classes = len(ds.sid2idx)

    train_idx, val_idx = [], []
    for sid, group in ds.df.groupby('student_id'):
        idxs = [i for i, it in enumerate(ds.items) if it[2] == int(sid)]  # indices in items
        cut = max(1, int(0.8 * len(idxs)))
        train_idx += idxs[:cut]
        val_idx   += idxs[cut:]

    subset_train = torch.utils.data.Subset(ds, train_idx)
    subset_val   = torch.utils.data.Subset(ds, val_idx)

    loader_train = DataLoader(subset_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    loader_val   = DataLoader(subset_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Model
    model = ArcFaceModel(num_classes=num_classes)
    model.to(device)

    if FREEZE_BACKBONE:
        for p in model.backbone.parameters():
            p.requires_grad = False

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ---- Train ----
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y, _ in tqdm(loader_train, desc=f"Epoch {epoch}/{EPOCHS}"):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x, y)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += x.size(0)
                loss_sum += loss.item() * x.size(0)
        train_acc = correct / max(1, total)
        train_loss = loss_sum / max(1, total)

        # ---- Validate ----
        model.eval()
        v_total, v_correct = 0, 0
        with torch.no_grad():
            for x, y, _ in loader_val:
                x = x.to(device); y = y.to(device)
                logits, _ = model(x, y)
                pred = logits.argmax(dim=1)
                v_correct += (pred == y).sum().item()
                v_total += x.size(0)
        val_acc = v_correct / max(1, v_total) if v_total else 0.0
        scheduler.step()

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        # Save best checkpoint
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            ckpt = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'sid2idx': ds.sid2idx,
                'config': {
                    'EPOCHS': EPOCHS, 'BATCH_SIZE': BATCH_SIZE, 'LR': LR,
                    'ARC_S': ARC_S, 'ARC_M': ARC_M, 'FREEZE_BACKBONE': FREEZE_BACKBONE
                }
            }
            torch.save(ckpt, OUT_DIR / 'ckpt_best.pth')
            print(f"[SAVE] Best checkpoint → {OUT_DIR / 'ckpt_best.pth'} (val_acc={val_acc:.4f})")

    # ---- Export prototypes & embeddings on the full dataset ----
    print("Computing prototypes & embeddings …")
    full_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Load best
    best = torch.load(OUT_DIR / 'ckpt_best.pth', map_location=device)
    model.load_state_dict(best['state_dict'])
    model.eval()

    # Prototypes
    protos = compute_prototypes(model.backbone, full_loader, device)
    with open(OUT_DIR / 'prototypes.json', 'w') as f:
        json.dump(protos, f, indent=2)

    # Embeddings CSV (auditing)
    rows = []
    with torch.no_grad():
        for x, y, sid in tqdm(full_loader, desc="Embeddings"):
            x = x.to(device)
            f = F.normalize(model.backbone.forward(x)).cpu().numpy()
            for i in range(f.shape[0]):
                rows.append({
                    'student_id': int(sid[i]),
                    'cls_idx': int(y[i].item()),
                    **{f'f{j}': float(f[i, j]) for j in range(f.shape[1])}
                })
    pd.DataFrame(rows).to_csv(OUT_DIR / 'embeddings.csv', index=False)
    print(f"[DONE] Artifacts written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
