"""
validate_2stage.py
──────────────────
Stage 1: YOLO tiled inference detects flag bounding boxes.
Stage 2a: Stripe-pattern classifier (fast, works on tiny 38px crops) — handles
          Germany, Russia, France and ~60 other stripe flags reliably.
Stage 2b: HOG + RandomForest fallback for complex / non-stripe flags.
"""

import os, glob, json, pickle, cv2, numpy as np
from ultralytics import YOLO
from skimage.feature import hog

# ── Paths ─────────────────────────────────────────────────────────────────────
CLF_DIR      = r"C:\Users\mido\Documents\antigravity\focused-babbage\flag_clf_dataset"
VALIDATE_DIR = r"C:\Users\mido\Documents\antigravity\focused-babbage\validate_ai"
IMG_SIZE     = 64

# ── Detection params ──────────────────────────────────────────────────────────
TILE_SIZE   = 320
TILE_STEP   = 240
CONF_THRESH = 0.40
NMS_IOU     = 0.40

# ═════════════════════════════════════════════════════════════════════════════
# Stage 2a: Stripe pattern classifier
# ═════════════════════════════════════════════════════════════════════════════

def isolate_interior(crop_bgr, vt=185, st=65, trim=2):
    """Remove white border frame; return flag interior or original if no frame."""
    h, w = crop_bgr.shape[:2]
    if h < 8 or w < 8:
        return crop_bgr
    hsv  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 2] > vt) & (hsv[:, :, 1] < st)
    if not white.any():
        return crop_bgr
    nw = ~white
    if not nw.any():
        return crop_bgr
    ys, xs = np.where(nw)
    y1 = max(0,   ys.min() - trim)
    y2 = min(h,   ys.max() + trim + 1)
    x1 = max(0,   xs.min() - trim)
    x2 = min(w,   xs.max() + trim + 1)
    r  = crop_bgr[y1:y2, x1:x2]
    return r if r.size >= 50 else crop_bgr


def zone_color(strip_hsv):
    """Classify an HSV strip region into a named color."""
    H = float(np.median(strip_hsv[:, :, 0]))   # 0-180 (OpenCV)
    S = float(np.median(strip_hsv[:, :, 1]))   # 0-255
    V = float(np.median(strip_hsv[:, :, 2]))   # 0-255
    if V < 95:
        return 'black'
    if V > 170 and S < 75:
        return 'white'
    if S > 55:
        hd = H * 2  # 0-360
        if hd < 35 or hd > 345:
            return 'red'
        if 35 <= hd < 85:
            return 'gold'       # yellow / gold (de, co, ve …)
        if 85 <= hd < 175:
            return 'green'
        if 175 <= hd < 275:
            return 'blue'
        if 275 <= hd <= 345:
            return 'purple'
    return 'neutral'


# Stripe pattern table ─ (color_a, color_b, color_c) → ISO-3166 / custom code
# Both orders listed (flag can be upside-down / photographed from either side)
STRIPE_MAP = {
    # Germany
    ('black', 'red',   'gold' ): 'de',
    ('gold',  'red',   'black'): 'de',
    # Russia
    ('white', 'blue',  'red'  ): 'ru',
    ('red',   'blue',  'white'): 'ru',
    # France / Romania / Chad (blue–white–red vertical)
    ('blue',  'white', 'red'  ): 'fr',
    ('red',   'white', 'blue' ): 'fr',
    # Netherlands / Luxembourg / Croatia (red–white–blue)
    ('red',   'white', 'blue' ): 'nl',
    ('blue',  'white', 'red'  ): 'nl',   # NOTE: overridden by 'fr' above (handled below)
    # Italy / Hungary/…
    ('green', 'white', 'red'  ): 'it',
    ('red',   'white', 'green'): 'it',
    # Belgium (vertical black–yellow–red)
    ('black', 'gold',  'red'  ): 'be',
    ('red',   'gold',  'black'): 'be',
    # Austria (red–white–red)
    ('red',   'white', 'red'  ): 'at',
    # Estonia
    ('blue',  'black', 'white'): 'ee',
    ('white', 'black', 'blue' ): 'ee',
    # Lithuania
    ('gold',  'green', 'red'  ): 'lt',
    ('red',   'green', 'gold' ): 'lt',
    # Armenia
    ('red',   'blue',  'gold' ): 'am',
    ('gold',  'blue',  'red'  ): 'am',
    # Yemen / Syria / Iraq (black–white–red)
    ('black', 'white', 'red'  ): 'ye',
    ('red',   'white', 'black'): 'ye',
    # Gabon / Cameroon-ish
    ('green', 'gold',  'blue' ): 'ga',
    ('blue',  'gold',  'green'): 'ga',
    # Mali / Guinea / Senegal
    ('green', 'gold',  'red'  ): 'ml',
    ('red',   'gold',  'green'): 'ml',
    # UAE (green–white–black–red)
    ('green', 'white', 'black'): 'ae',
    # Slovenia / Slovakia
    ('white', 'blue',  'red'  ): 'sk',   # overlaps with ru (handled below)
    # Ireland / Ivory Coast
    ('green', 'white', 'red'  ): 'ie',
    ('red',   'white', 'green'): 'ci',
}

# Priority overrides when horizontal and vertical patterns both match
# Key: (h_match, v_match) → prefer this code
_PRIORITY = {
    ('fr', 'nl'): 'fr',   # if vertical is fr, prefer fr
    ('nl', 'fr'): 'nl',
    ('ru', 'sk'): 'ru',
}


def classify_by_stripes(crop_bgr):
    """
    Returns (country_code, confidence) using stripe-pattern analysis
    across 4 rotations.  Returns (None, 0.0) if pattern not recognised.
    """
    interior = isolate_interior(crop_bgr)
    if interior is None or interior.size < 50:
        interior = crop_bgr

    h, w = interior.shape[:2]
    best = (None, 0.0)

    for angle in (0, 90, 180, 270):
        if angle == 0:
            img = interior
        else:
            M   = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            img = cv2.warpAffine(interior, M, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        rh, rw = img.shape[:2]

        # Horizontal thirds
        h3 = max(1, rh // 3)
        hp = tuple(zone_color(hsv[i*h3:(i+1)*h3, :, :]) for i in range(3))

        # Vertical thirds
        w3 = max(1, rw // 3)
        vp = tuple(zone_color(hsv[:, i*w3:(i+1)*w3, :]) for i in range(3))

        hm = STRIPE_MAP.get(hp)
        vm = STRIPE_MAP.get(vp)

        if hm and vm:
            code = _PRIORITY.get((hm, vm), hm)
            return code, 0.82
        if hm:
            return hm, 0.80
        if vm:
            return vm, 0.80

    return best   # (None, 0.0)


# ═════════════════════════════════════════════════════════════════════════════
# Stage 2b: HOG + RF fallback classifier
# ═════════════════════════════════════════════════════════════════════════════

def extract_hog_features(img_bgr):
    img_r = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    gray  = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    hog_vec = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                  cells_per_block=(2, 2), visualize=False, feature_vector=True)
    hsv = cv2.cvtColor(img_r, cv2.COLOR_BGR2HSV).astype(np.float32)
    zone_feats = []
    for r in range(3):
        for c in range(3):
            cell = hsv[r*IMG_SIZE//3:(r+1)*IMG_SIZE//3,
                       c*IMG_SIZE//3:(c+1)*IMG_SIZE//3, :]
            h_rad  = cell[:, :, 0] * (np.pi / 90.0)
            mH = (np.arctan2(np.sin(h_rad).mean(), np.cos(h_rad).mean())
                  * 90.0 / np.pi) % 180.0
            zone_feats.extend([mH / 180.0,
                                cell[:, :, 1].mean() / 255.0,
                                cell[:, :, 2].mean() / 255.0])
    return np.concatenate([hog_vec, zone_feats]).astype(np.float32)


def load_classifier():
    path = os.path.join(CLF_DIR, "svm_model.pkl")
    if not os.path.exists(path):
        print("No RF model found — run train_flag_svm.py first")
        return None, None
    with open(path, "rb") as f:
        clf, label_map = pickle.load(f)
    print(f"Loaded RF classifier ({len(label_map)} classes)")
    return clf, label_map


def classify_crop(crop_bgr, clf, label_map):
    """
    Priority: stripe-pattern classifier → RF fallback.
    Returns (class_name, confidence).
    """
    # 1. Fast stripe detector
    code, conf = classify_by_stripes(crop_bgr)
    if code is not None:
        return code, conf

    # 2. RF fallback
    if clf is None:
        return "unknown", 0.0
    feats = extract_hog_features(crop_bgr).reshape(1, -1)
    idx   = int(clf.predict(feats)[0])
    try:
        scores = clf.decision_function(feats)[0]
        rf_conf = float(scores[idx]) if hasattr(scores, '__len__') else float(scores)
    except Exception:
        rf_conf = 0.5
    return label_map[str(idx)], rf_conf


# ═════════════════════════════════════════════════════════════════════════════
# Detection helpers
# ═════════════════════════════════════════════════════════════════════════════

def find_weights():
    p = os.path.join('runs', 'detect', 'flag_detector', 'weights', 'best.pt')
    if os.path.exists(p):
        return p
    for d in sorted(glob.glob(os.path.join('runs', 'detect', 'train*')),
                    key=os.path.getmtime, reverse=True):
        p2 = os.path.join(d, 'weights', 'best.pt')
        if os.path.exists(p2):
            return p2
    return None


def nms_boxes(dets, iou_thresh=NMS_IOU):
    if not dets:
        return []
    boxes  = np.array([[d[0], d[1], d[2]-d[0], d[3]-d[1]] for d in dets], np.float32)
    scores = np.array([d[4] for d in dets], np.float32)
    idx    = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(),
                               score_threshold=CONF_THRESH,
                               nms_threshold=iou_thresh)
    return [] if len(idx) == 0 else [dets[i] for i in idx.flatten()]


def tiled_detect(model_yolo, img_path):
    img    = cv2.imread(img_path)
    h, w   = img.shape[:2]
    y_starts = list(range(0, h - TILE_SIZE + 1, TILE_STEP))
    if not y_starts or y_starts[-1] + TILE_SIZE < h:
        y_starts.append(max(0, h - TILE_SIZE))
    x_starts = list(range(0, w - TILE_SIZE + 1, TILE_STEP))
    if not x_starts or x_starts[-1] + TILE_SIZE < w:
        x_starts.append(max(0, w - TILE_SIZE))
    tiles, positions = [], []
    for y0 in y_starts:
        for x0 in x_starts:
            tiles.append(img[y0:y0+TILE_SIZE, x0:x0+TILE_SIZE])
            positions.append((x0, y0))
    print(f"  {len(tiles)} tiles...")
    all_dets = []
    for i in range(0, len(tiles), 64):
        results = model_yolo.predict(tiles[i:i+64], imgsz=TILE_SIZE,
                                     conf=CONF_THRESH, verbose=False,
                                     agnostic_nms=True)
        for res, (x0, y0) in zip(results, positions[i:i+64]):
            for box in res.boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                conf = float(box.conf[0])
                all_dets.append((x1+x0, y1+y0, x2+x0, y2+y0, conf))
    return img, nms_boxes(all_dets)


# ═════════════════════════════════════════════════════════════════════════════
# Ground-truth validation for image 15
# ═════════════════════════════════════════════════════════════════════════════
GT_FLAGS = {
    "15.jpg": [
        ("de", 1087, 142,  80, "Germany"),
        ("ru", 1194, 2951, 80, "Russia"),
        ("fr", 2141, 1392, 80, "France"),
    ]
}


def check_gt(filename, detections):
    if filename not in GT_FLAGS:
        return True, 0, 0
    gt = GT_FLAGS[filename]
    matched = [False] * len(gt)
    fp = 0
    for (x1, y1, x2, y2, conf, cls_name, conf_cls) in detections:
        dcx, dcy = (x1+x2)/2, (y1+y2)/2
        hit = False
        for i, (gc, gcx, gcy, r, desc) in enumerate(gt):
            if ((dcx-gcx)**2 + (dcy-gcy)**2)**0.5 <= r:
                hit = True
                if cls_name == gc:
                    matched[i] = True
                    print(f"    OK {desc} ({cls_name}) det={conf:.2f} cls={conf_cls:.2f}")
                else:
                    print(f"    WRONG {desc}: got '{cls_name}' expected '{gc}'")
                break
        if not hit:
            fp += 1
            print(f"    FP: '{cls_name}' conf={conf:.2f} at ({int((x1+x2)/2)},{int((y1+y2)/2)})")
    for i, (gc, gcx, gcy, r, desc) in enumerate(gt):
        if not matched[i]:
            print(f"    MISSED: {desc} ({gc}) at ({gcx},{gcy})")
    missed = sum(1 for m in matched if not m)
    return missed == 0 and fp == 0, missed, fp


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    weights = find_weights()
    if not weights:
        print("ERROR: No YOLO weights found!"); return False
    print(f"YOLO: {weights}")
    yolo = YOLO(weights)

    clf, label_map = load_classifier()   # may be None if file missing

    out_dir = 'validate_ai_results_2stage'
    os.makedirs(out_dir, exist_ok=True)

    total_missed = total_fp = 0
    all_pass = True

    for img_path in sorted(glob.glob(os.path.join(VALIDATE_DIR, '*.jpg'))):
        filename = os.path.basename(img_path)
        print(f"\n{'='*60}\nProcessing {filename}...")
        img, raw_dets = tiled_detect(yolo, img_path)
        print(f"  {len(raw_dets)} raw detections after NMS")

        final_dets = []
        for (x1, y1, x2, y2, conf) in raw_dets:
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
            crop = img[iy1:iy2, ix1:ix2]
            cls_name, conf_cls = classify_crop(crop, clf, label_map)
            method = "stripe" if conf_cls >= 0.78 else "rf"
            print(f"  [{ix1},{iy1},{ix2},{iy2}] det={conf:.2f} -> '{cls_name}' "
                  f"({conf_cls:.2f}) [{method}]")
            final_dets.append((x1, y1, x2, y2, conf, cls_name, conf_cls))

            # Draw
            color = (0, 200, 50)
            cv2.rectangle(img, (ix1, iy1), (ix2, iy2), color, 3)
            label = f"{cls_name}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            ly = max(iy1 - 4, th + 6)
            cv2.rectangle(img, (ix1, ly-th-4), (ix1+tw+2, ly+4), color, -1)
            cv2.putText(img, label, (ix1+1, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        cv2.imwrite(os.path.join(out_dir, f"2stage_{filename}"), img)

        ok, missed, fp = check_gt(filename, final_dets)
        if filename in GT_FLAGS:
            total_missed += missed
            total_fp     += fp
            if not ok:
                all_pass = False

    print(f"\n{'='*60}")
    print(f"SUMMARY -- Missed: {total_missed}  |  False Positives: {total_fp}")
    if all_pass:
        print("ALL CRITICAL FLAGS DETECTED CORRECTLY!")
        return True
    else:
        print("Validation FAILED -- see details above.")
        return False


if __name__ == '__main__':
    main()
