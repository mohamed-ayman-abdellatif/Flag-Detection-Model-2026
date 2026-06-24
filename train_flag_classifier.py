"""
train_flag_classifier.py  (in-memory augmentation — no disk I/O)
─────────────────────────────────────────────────────────────────
Loads all 319 reference flags into RAM, then generates augmented
"aerial crop" samples on-the-fly inside PyTorch Dataset.__getitem__.
No temporary files are written to disk.

GPU-accelerated training.  Typical wall time: ~5-8 minutes.

Saves:
  flag_clf_dataset/best_model.pt   ← best val-accuracy state dict
  flag_clf_dataset/label_map.json  ← {str(idx): class_name}
"""

import os, glob, json, random, time
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ── Config ─────────────────────────────────────────────────────────────────────
REF_COUNTRY  = r"C:\Users\mido\Documents\antigravity\focused-babbage\reference\country"
REF_INST     = r"C:\Users\mido\Documents\antigravity\focused-babbage\reference\institution"
OUT_DIR      = r"C:\Users\mido\Documents\antigravity\focused-babbage\flag_clf_dataset"

IMG_SIZE     = 64
N_AUG_TRAIN  = 60      # virtual augmented samples per class per epoch
N_AUG_VAL    = 15
BATCH_SIZE   = 128
EPOCHS       = 40
LR           = 4e-4
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUT_DIR, exist_ok=True)
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Load reference images into RAM ─────────────────────────────────────────────
def load_references():
    refs = {}
    for path in sorted(glob.glob(os.path.join(REF_COUNTRY, "*.png"))):
        n = os.path.splitext(os.path.basename(path))[0]
        img = cv2.imread(path)
        if img is not None:
            # Pre-resize to 128×128 — augmentation runs ~350× faster than on 2560×1536
            refs[n] = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    for path in sorted(glob.glob(os.path.join(REF_INST, "*"))):
        ext = path.lower().rsplit('.', 1)[-1]
        if ext not in ('png', 'jpg', 'jpeg', 'gif'):
            continue
        try:
            pil = Image.open(path).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            n   = os.path.splitext(os.path.basename(path))[0]
            if n not in refs:
                refs[n] = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        except Exception:
            pass
    return refs

print("Loading reference flags into RAM…")
refs   = load_references()
CLASSES = sorted(refs.keys())
N_CLS   = len(CLASSES)
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for i, c in enumerate(CLASSES)}
print(f"Loaded {N_CLS} flag classes")

os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "label_map.json"), "w") as f:
    json.dump({str(i): c for i, c in IDX2CLS.items()}, f, indent=2)
with open(os.path.join(OUT_DIR, "classes.txt"), "w") as f:
    f.write("\n".join(CLASSES))

# ── In-memory augmentation ─────────────────────────────────────────────────────
def augment_bgr(img, size=IMG_SIZE, is_val=False):
    """Apply aerial-photography augmentation to a reference flag image (BGR)."""
    h, w = img.shape[:2]

    if not is_val:
        # 1. Random rotation
        angle = random.uniform(0, 360)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_WRAP)

        # 2. Perspective warp
        margin = random.uniform(0.02, 0.12)
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pert  = np.random.uniform(-margin, margin, (4, 2)).astype(np.float32)
        pert *= [w, h]
        pts2  = pts1 + pert
        Mw = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, Mw, (w, h), borderMode=cv2.BORDER_WRAP)

        # 3. Random crop + zoom
        frac = random.uniform(0.55, 1.0)
        ch, cw = max(8, int(h * frac)), max(8, int(w * frac))
        y0 = random.randint(0, h - ch)
        x0 = random.randint(0, w - cw)
        img = img[y0:y0+ch, x0:x0+cw]

    # 4. Gaussian blur (simulate aerial distance)
    sigma = random.uniform(0.5, 2.8) if not is_val else 1.5
    k = max(3, int(sigma * 3) | 1)
    img = cv2.GaussianBlur(img, (k, k), sigma)

    # 5. HSV: desaturate + brightness jitter
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat_f = random.uniform(0.25, 0.80) if not is_val else 0.55
    val_f = random.uniform(0.70, 1.30) if not is_val else 1.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_f, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_f, 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 6. Gaussian noise
    if not is_val:
        noise = np.random.normal(0, random.uniform(3, 12), img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 7. Optional white border (simulate physical flag card frame)
    if random.random() < (0.70 if not is_val else 0.50):
        bw = random.randint(2, 8) if not is_val else 4
        img = cv2.copyMakeBorder(img, bw, bw, bw, bw,
                                  cv2.BORDER_CONSTANT, value=(210, 210, 210))

    # 8. Resize to model input
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    return img

# ── PyTorch Dataset (on-the-fly augmentation) ─────────────────────────────────
class FlagDataset(Dataset):
    def __init__(self, refs, classes, cls2idx, n_aug, is_val=False):
        self.items = [(refs[c], cls2idx[c]) for c in classes for _ in range(n_aug)]
        random.shuffle(self.items)
        self.is_val = is_val
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_bgr, label = self.items[idx]
        aug = augment_bgr(img_bgr, is_val=self.is_val)
        # BGR → RGB for PIL → tensor
        aug_rgb = cv2.cvtColor(aug, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(aug_rgb)
        return self.to_tensor(pil), label

# ── FlagCNN ────────────────────────────────────────────────────────────────────
class FlagCNN(nn.Module):
    def __init__(self, n_classes, img_size=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 64→32

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 32→16

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 16→8

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 8→4
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.50),
            nn.Linear(256, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ── Build datasets & loaders ──────────────────────────────────────────────────
print(f"\nBuilding datasets (in-memory, {N_AUG_TRAIN} train + {N_AUG_VAL} val per class)…")
train_ds = FlagDataset(refs, CLASSES, CLS2IDX, N_AUG_TRAIN, is_val=False)
val_ds   = FlagDataset(refs, CLASSES, CLS2IDX, N_AUG_VAL,   is_val=True)
print(f"Train: {len(train_ds):,} samples | Val: {len(val_ds):,} samples")

# num_workers=0 avoids multiprocessing pickling issues on Windows
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                      num_workers=0, pin_memory=(DEVICE.type == "cuda"))
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=0, pin_memory=(DEVICE.type == "cuda"))

# ── Train ──────────────────────────────────────────────────────────────────────
model = FlagCNN(N_CLS, IMG_SIZE).to(DEVICE)
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR, steps_per_epoch=len(train_dl), epochs=EPOCHS)

best_val_acc = 0.0
best_path    = os.path.join(OUT_DIR, "best_model.pt")

print(f"\nStarting training ({EPOCHS} epochs)…")
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()

    # ---- train ----
    model.train()
    tr_loss = tr_correct = tr_total = 0
    for imgs, labels in train_dl:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        tr_loss    += loss.item() * len(imgs)
        tr_correct += (logits.detach().argmax(1) == labels).sum().item()
        tr_total   += len(imgs)

    # ---- val ----
    model.eval()
    val_correct = val_total = 0
    with torch.no_grad():
        for imgs, labels in val_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(1)
            val_correct += (preds == labels).sum().item()
            val_total   += len(imgs)

    val_acc = val_correct / val_total * 100
    elapsed = time.time() - t0
    print(f"Epoch {epoch:02d}/{EPOCHS}  "
          f"loss={tr_loss/tr_total:.3f}  "
          f"val_acc={val_acc:.1f}%  ({elapsed:.0f}s)")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_path)
        print(f"  ★ Best so far: {best_val_acc:.1f}% — saved")

print(f"\n✅ Done. Best val acc = {best_val_acc:.1f}%")
print(f"   Model → {best_path}")
