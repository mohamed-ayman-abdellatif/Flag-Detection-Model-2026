"""
train_flag_svm.py
─────────────────
Trains a RandomForest on HOG + color zone features.

• HOG features: capture stripe orientation (vertical French vs horizontal German)
• Color zone features: 3x3 spatial HSV grid distinguishes de vs ao by color layout
• RandomForest: non-linear, handles 319 classes well with minimal samples
• Training: ~3 min on CPU for 319 x 30 = 9,570 samples

Saves:
  flag_clf_dataset/svm_model.pkl    <- trained RandomForest pipeline
  flag_clf_dataset/label_map.json  <- {str(idx): class_name}
"""

import os, glob, json, random, time, pickle
import cv2
import numpy as np
from PIL import Image

from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ── Paths ─────────────────────────────────────────────────────────────────────
REF_COUNTRY = r"C:\Users\mido\Documents\antigravity\focused-babbage\reference\country"
REF_INST    = r"C:\Users\mido\Documents\antigravity\focused-babbage\reference\institution"
OUT_DIR     = r"C:\Users\mido\Documents\antigravity\focused-babbage\flag_clf_dataset"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE   = 64     # resize target for feature extraction
N_AUG      = 30    # 30 x 319 = 9,570 samples
SEED       = 42
random.seed(SEED); np.random.seed(SEED)

# ── Load references pre-resized to 128×128 ────────────────────────────────────
def load_references():
    refs = {}
    for path in sorted(glob.glob(os.path.join(REF_COUNTRY, "*.png"))):
        n = os.path.splitext(os.path.basename(path))[0]
        img = cv2.imread(path)
        if img is not None:
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

# ── Augmentation (same as CNN pipeline) ──────────────────────────────────────
def augment(img, size=IMG_SIZE):
    h, w = img.shape[:2]
    # rotation
    angle = random.uniform(0, 360)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_WRAP)
    # perspective
    margin = random.uniform(0.02, 0.10)
    pts1 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    pert = (np.random.uniform(-margin, margin, (4,2)) * [w, h]).astype(np.float32)
    Mw = cv2.getPerspectiveTransform(pts1, pts1 + pert)
    img = cv2.warpPerspective(img, Mw, (w, h), borderMode=cv2.BORDER_WRAP)
    # random crop
    frac = random.uniform(0.60, 1.0)
    ch, cw = max(8, int(h*frac)), max(8, int(w*frac))
    y0, x0 = random.randint(0, h-ch), random.randint(0, w-cw)
    img = img[y0:y0+ch, x0:x0+cw]
    # blur -- keep sigma low to preserve HOG edge features
    sigma = random.uniform(0.3, 1.5)
    k = max(3, int(sigma*3)|1)
    img = cv2.GaussianBlur(img, (k, k), sigma)
    # desaturate + brightness
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] *= random.uniform(0.25, 0.80)
    hsv[:,:,2] *= random.uniform(0.70, 1.30)
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    # noise
    img = np.clip(img.astype(np.float32) + np.random.normal(0, random.uniform(2,10), img.shape),
                  0, 255).astype(np.uint8)
    # optional white border
    if random.random() < 0.65:
        bw = random.randint(2, 7)
        img = cv2.copyMakeBorder(img, bw, bw, bw, bw, cv2.BORDER_CONSTANT, value=(210,210,210))
    # resize
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)

# ── Feature extraction: HOG + 3×3 zone HSV stats ─────────────────────────────
def extract_features(img_bgr):
    """
    Returns a 1-D float32 feature vector combining:
    • HOG on grayscale (captures stripe orientation & spacing)
    • Mean H, S, V per 3×3 spatial zone (captures color palette per region)
    """
    # HOG
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hog_vec = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                  cells_per_block=(2, 2), visualize=False, feature_vector=True)

    # Spatial color zones (3×3 grid)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_size, w_size = img_bgr.shape[:2]
    zone_feats = []
    for r in range(3):
        for c in range(3):
            cell = hsv[r*h_size//3:(r+1)*h_size//3,
                       c*w_size//3:(c+1)*w_size//3, :]
            # Circular mean of hue to handle red wraparound (H=0 ≈ H=180)
            h_rad = cell[:, :, 0] * (np.pi / 90.0)
            mean_H = (np.arctan2(np.sin(h_rad).mean(), np.cos(h_rad).mean())
                      * 90.0 / np.pi) % 180.0
            mean_S = cell[:, :, 1].mean() / 255.0
            mean_V = cell[:, :, 2].mean() / 255.0
            zone_feats.extend([mean_H / 180.0, mean_S, mean_V])

    return np.concatenate([hog_vec, zone_feats]).astype(np.float32)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading reference flags…")
    refs    = load_references()
    classes = sorted(refs.keys())
    cls2idx = {c: i for i, c in enumerate(classes)}
    idx2cls = {str(i): c for i, c in enumerate(classes)}
    print(f"Loaded {len(classes)} classes")

    with open(os.path.join(OUT_DIR, "label_map.json"), "w") as f:
        json.dump(idx2cls, f, indent=2)
    with open(os.path.join(OUT_DIR, "classes.txt"), "w") as f:
        f.write("\n".join(classes))

    # Build augmented feature matrix
    print(f"\nGenerating {N_AUG} augmented samples × {len(classes)} classes = "
          f"{N_AUG * len(classes):,} total…")
    t0 = time.time()
    X, y = [], []
    for i, name in enumerate(classes):
        img = refs[name]
        for _ in range(N_AUG):
            aug   = augment(img)
            feats = extract_features(aug)
            X.append(feats)
            y.append(cls2idx[name])
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(classes)} classes augmented…")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"Feature matrix: {X.shape}  (took {time.time()-t0:.1f}s)")

    # Train RandomForest (non-linear, much better than linear SVM for 319 classes)
    print("\nTraining RandomForest...")
    t1 = time.time()
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=1,
            n_jobs=-1, random_state=SEED, class_weight='balanced'
        )),
    ])
    clf.fit(X, y)
    print(f"Training done in {time.time()-t1:.1f}s")


    # Validation accuracy (on last 10 aug per class)
    val_correct = val_total = 0
    for i, name in enumerate(classes):
        img = refs[name]
        for _ in range(10):
            aug   = augment(img)
            feats = extract_features(aug).reshape(1, -1)
            pred  = clf.predict(feats)[0]
            if pred == cls2idx[name]:
                val_correct += 1
            val_total += 1

    val_acc = val_correct / val_total * 100
    print(f"Validation accuracy: {val_acc:.1f}%  ({val_correct}/{val_total})")

    # Save model
    model_path = os.path.join(OUT_DIR, "svm_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump((clf, idx2cls), f)
    print(f"\nSaved RF model -> {model_path}")
    return val_acc

if __name__ == "__main__":
    main()
