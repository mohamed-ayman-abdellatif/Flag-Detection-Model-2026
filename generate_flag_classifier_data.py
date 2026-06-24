"""
generate_flag_classifier_data.py
──────────────────────────────────
Creates an augmented training/validation dataset for the flag classifier.

For each of the 319 reference flags, generates N_AUG synthetic "aerial crop"
versions by applying:
  • Random rotation 0–360°           (flags can be placed in any orientation)
  • Perspective warp                  (viewed from a drone at an angle)
  • Gaussian blur  σ 0.8–3.0          (flag is 25–45 px in the real image)
  • Saturation reduction 30–75%       (aerial haze)
  • Brightness jitter ±20%
  • Gaussian noise                    (sensor noise)
  • Optional white border frame       (physical flag has a white card border)

Output structure (YOLO-style, but for classification):
  flag_clf_dataset/
    train/<class_idx>/   ← one sub-folder per flag class
    val/<class_idx>/
  classes.txt            ← ordered list of flag names
"""

import os, glob, random, json
import cv2
import numpy as np
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
REF_COUNTRY  = r"C:\Users\mido\Documents\antigravity\focused-babbage\reference\country"
REF_INST     = r"C:\Users\mido\Documents\antigravity\focused-babbage\reference\institution"
OUT_DIR      = r"C:\Users\mido\Documents\antigravity\focused-babbage\flag_clf_dataset"
IMG_SIZE     = 64           # model input size (64×64)
N_AUG_TRAIN  = 300          # augmented samples per class for train
N_AUG_VAL    = 60           # augmented samples per class for val
SEED         = 42

random.seed(SEED)
np.random.seed(SEED)

# ── Load all reference images ──────────────────────────────────────────────────
def load_references():
    refs = {}
    for path in sorted(glob.glob(os.path.join(REF_COUNTRY, "*.png"))):
        n = os.path.splitext(os.path.basename(path))[0]
        img = cv2.imread(path)
        if img is not None:
            refs[n] = img
    for path in sorted(glob.glob(os.path.join(REF_INST, "*"))):
        ext = path.lower().rsplit('.', 1)[-1]
        if ext not in ('png', 'jpg', 'jpeg', 'gif'):
            continue
        try:
            pil = Image.open(path).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            n   = os.path.splitext(os.path.basename(path))[0]
            if n not in refs:
                refs[n] = img
        except Exception:
            pass
    return refs

# ── Augmentation pipeline ──────────────────────────────────────────────────────
def augment(img, size=IMG_SIZE):
    h, w = img.shape[:2]

    # 1. Random rotation
    angle = random.uniform(0, 360)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_WRAP)

    # 2. Perspective warp (subtle)
    margin = random.uniform(0.03, 0.12)
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    noise = np.random.uniform(-margin, margin, (4, 2)) * np.array([[w, h]])
    pts2  = (pts1 + noise).astype(np.float32)
    Mwarp = cv2.getPerspectiveTransform(pts1, pts2)
    img   = cv2.warpPerspective(img, Mwarp, (w, h), borderMode=cv2.BORDER_WRAP)

    # 3. Blur (simulate aerial distance)
    sigma = random.uniform(0.8, 3.0)
    k     = max(3, int(sigma * 3) | 1)   # odd kernel
    img   = cv2.GaussianBlur(img, (k, k), sigma)

    # 4. HSV jitter: desaturate + brightness
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= random.uniform(0.30, 0.80)   # desaturate
    hsv[:, :, 2] *= random.uniform(0.75, 1.25)    # brightness
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 5. Gaussian noise
    noise_img = np.random.normal(0, random.uniform(2, 10), img.shape)
    img = np.clip(img.astype(np.float32) + noise_img, 0, 255).astype(np.uint8)

    # 6. Random crop + zoom
    crop_frac = random.uniform(0.60, 1.0)
    ch, cw = int(h * crop_frac), int(w * crop_frac)
    y0 = random.randint(0, h - ch)
    x0 = random.randint(0, w - cw)
    img = img[y0:y0+ch, x0:x0+cw]

    # 7. Resize to target
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)

    # 8. Optional white border (simulate physical flag card frame)
    if random.random() < 0.70:
        bw = random.randint(2, 8)
        img = cv2.copyMakeBorder(img, bw, bw, bw, bw,
                                  cv2.BORDER_CONSTANT, value=(210, 210, 210))
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)

    return img

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading reference flags …")
    refs = load_references()
    classes = sorted(refs.keys())
    print(f"Found {len(classes)} flag classes")

    # Save classes list
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "classes.txt"), "w") as f:
        f.write("\n".join(classes))
    with open(os.path.join(OUT_DIR, "class_to_idx.json"), "w") as f:
        json.dump({c: i for i, c in enumerate(classes)}, f, indent=2)

    for split, n_aug in [("train", N_AUG_TRAIN), ("val", N_AUG_VAL)]:
        print(f"\nGenerating {split} split ({n_aug} augmentations × {len(classes)} classes) …")
        for idx, name in enumerate(classes):
            img = refs[name]
            out_class_dir = os.path.join(OUT_DIR, split, str(idx))
            os.makedirs(out_class_dir, exist_ok=True)
            for i in range(n_aug):
                aug = augment(img)
                cv2.imwrite(os.path.join(out_class_dir, f"{i:04d}.jpg"), aug,
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
            if (idx + 1) % 50 == 0:
                print(f"  {idx+1}/{len(classes)} classes done")

    total_train = len(classes) * N_AUG_TRAIN
    total_val   = len(classes) * N_AUG_VAL
    print(f"\n✅ Dataset created in {OUT_DIR}")
    print(f"   Train: {total_train} images | Val: {total_val} images")
    print(f"   Classes: {len(classes)}")

if __name__ == "__main__":
    main()
