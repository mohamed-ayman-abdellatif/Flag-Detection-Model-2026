"""
Tiled validation script for flag detection.
Tiles the full-resolution 4056x3840 GoPro images into 320x320 tiles,
runs YOLOv8 inference on each tile at imgsz=320 (matching training resolution),
maps detections back to original image coordinates, then applies global NMS.
"""

import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------
# GROUND TRUTH for automated pass/fail scoring
# Each entry: (class_name, cx, cy, radius_px, description)
# radius_px: how close a detection center must be to count as a match
# ---------------------------------------------------------------
GT_FLAGS = {
    "15.jpg": [
        ("de", 1087, 142,  80, "Germany"),
        ("ru", 1194, 2951, 80, "Russia"),
        ("fr", 2141, 1392, 80, "France"),
    ]
}

TILE_SIZE = 320
TILE_STEP = 240   # 80-pixel overlap so nothing falls between tiles
CONF_THRESH = 0.10
NMS_IOU_THRESH = 0.30


def find_best_weights():
    runs_dir = 'runs/detect'
    if not os.path.exists(runs_dir):
        return None
    train_dirs = glob.glob(os.path.join(runs_dir, 'train*'))
    if not train_dirs:
        return None
    train_dirs.sort(key=os.path.getmtime, reverse=True)
    for d in train_dirs:
        p = os.path.join(d, 'weights', 'best.pt')
        if os.path.exists(p):
            return p
    return None


def global_nms(detections, iou_thresh=NMS_IOU_THRESH):
    """Class-agnostic NMS over list of (x1,y1,x2,y2,conf,cls) tuples."""
    if not detections:
        return []
    boxes  = np.array([[d[0], d[1], d[2], d[3]] for d in detections], dtype=np.float32)
    scores = np.array([d[4] for d in detections], dtype=np.float32)
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(),
        scores.tolist(),
        score_threshold=CONF_THRESH,
        nms_threshold=iou_thresh
    )
    if len(indices) == 0:
        return []
    indices = indices.flatten()
    return [detections[i] for i in indices]


def run_tiled_inference(model, img_path):
    """Return list of (x1,y1,x2,y2,conf,cls_name) in original image coords."""
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    all_dets = []

    # Generate tile positions with TILE_STEP stride
    y_starts = list(range(0, h - TILE_SIZE + 1, TILE_STEP))
    if not y_starts or y_starts[-1] + TILE_SIZE < h:
        y_starts.append(max(0, h - TILE_SIZE))

    x_starts = list(range(0, w - TILE_SIZE + 1, TILE_STEP))
    if not x_starts or x_starts[-1] + TILE_SIZE < w:
        x_starts.append(max(0, w - TILE_SIZE))

    tiles, positions = [], []
    for y0 in y_starts:
        for x0 in x_starts:
            tile = img[y0:y0+TILE_SIZE, x0:x0+TILE_SIZE]
            tiles.append(tile)
            positions.append((x0, y0))

    print(f"  → Running on {len(tiles)} tiles...")

    BATCH = 64
    for i in range(0, len(tiles), BATCH):
        batch_tiles = tiles[i:i+BATCH]
        batch_pos   = positions[i:i+BATCH]
        results = model.predict(
            batch_tiles,
            imgsz=TILE_SIZE,
            conf=CONF_THRESH,
            verbose=False,
            agnostic_nms=True
        )
        for res, (x0, y0) in zip(results, batch_pos):
            for box in res.boxes:
                bx1, by1, bx2, by2 = map(float, box.xyxy[0])
                conf = float(box.conf[0])
                cls_name = model.names[int(box.cls[0])]
                all_dets.append((
                    bx1 + x0, by1 + y0,
                    bx2 + x0, by2 + y0,
                    conf, cls_name
                ))

    # Global NMS
    dets = global_nms(all_dets)
    return img, dets


def draw_detections(img, detections):
    for (x1, y1, x2, y2, conf, cls_name) in detections:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(img, label, (x1, max(y1-12, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return img


def check_gt(filename, detections):
    if filename not in GT_FLAGS:
        return True, 0, 0          # no GT → always pass
    gt_list = GT_FLAGS[filename]
    matched_gt = [False] * len(gt_list)
    false_positives = 0

    for (x1, y1, x2, y2, conf, cls_name) in detections:
        dcx = (x1 + x2) / 2
        dcy = (y1 + y2) / 2
        hit_gt = False
        for idx, (gt_cls, gt_cx, gt_cy, radius, desc) in enumerate(gt_list):
            dist = ((dcx - gt_cx)**2 + (dcy - gt_cy)**2) ** 0.5
            if dist <= radius:
                hit_gt = True
                if cls_name == gt_cls:
                    matched_gt[idx] = True
                    print(f"    ✅ CORRECT: {desc} ({cls_name}) conf={conf:.2f} dist={dist:.0f}px")
                else:
                    print(f"    ⚠️  WRONG CLASS at {desc}: got '{cls_name}' expected '{gt_cls}' conf={conf:.2f}")
                break
        if not hit_gt:
            false_positives += 1
            print(f"    ❌ FALSE POS: '{cls_name}' conf={conf:.2f} at ({int((x1+x2)/2)},{int((y1+y2)/2)})")

    missed = 0
    for idx, (gt_cls, gt_cx, gt_cy, radius, desc) in enumerate(gt_list):
        if not matched_gt[idx]:
            missed += 1
            print(f"    💀 MISSED: {desc} ({gt_cls}) at ({gt_cx},{gt_cy})")

    success = (missed == 0) and (false_positives == 0)
    return success, missed, false_positives


def main():
    weights = find_best_weights()
    if not weights:
        print("ERROR: No trained weights found. Run train_flag_yolo.py first.")
        return False
    print(f"Loading weights: {weights}")
    model = YOLO(weights)

    out_dir = 'validate_ai_results_tiled320'
    os.makedirs(out_dir, exist_ok=True)

    total_missed = 0
    total_fp     = 0
    all_pass     = True

    for img_path in sorted(glob.glob('validate_ai/*.jpg')):
        filename = os.path.basename(img_path)
        print(f"\n{'='*60}")
        print(f"Processing {filename}...")
        img, dets = run_tiled_inference(model, img_path)
        print(f"  → {len(dets)} detections after NMS")
        for d in dets:
            print(f"    {d[5]} conf={d[4]:.2f} at [{int(d[0])},{int(d[1])},{int(d[2])},{int(d[3])}]")

        ok, missed, fp = check_gt(filename, dets)
        if filename in GT_FLAGS:
            total_missed += missed
            total_fp     += fp
            if not ok:
                all_pass = False

        img = draw_detections(img, dets)
        cv2.imwrite(os.path.join(out_dir, f"tiled320_{filename}"), img)

    print(f"\n{'='*60}")
    print(f"SUMMARY — Missed: {total_missed}  |  False Positives: {total_fp}")
    if all_pass:
        print("🏆 ALL CRITICAL FLAGS DETECTED CORRECTLY!")
        return True
    else:
        print("❌ Validation FAILED — see details above.")
        return False


if __name__ == '__main__':
    main()
