import os
import glob
import cv2
import json
from ultralytics import YOLO

# Known ground truth locations of flags in validation images
GT_FLAGS = {
    "15.jpg": [
        {"cls": "de", "cx": 1087, "cy": 142, "desc": "Germany"},
        {"cls": "ru", "cx": 1194, "cy": 2951, "desc": "Russia"},
        {"cls": "fr", "cx": 2141, "cy": 1392, "desc": "France"}
    ]
}

def check_detections(img_name, detections):
    if img_name not in GT_FLAGS:
        return True, 0, 0
    
    gt_list = GT_FLAGS[img_name]
    detected_gt = [False] * len(gt_list)
    false_positives = 0
    
    for det in detections:
        x1, y1, x2, y2, conf, cls_name = det
        dcx = (x1 + x2) / 2
        dcy = (y1 + y2) / 2
        
        # Check if it matches any GT flag
        matched = False
        for idx, gt in enumerate(gt_list):
            # Check center distance
            dist = ((dcx - gt["cx"])**2 + (dcy - gt["cy"])**2)**0.5
            if dist < 80: # 80 pixels radius
                matched = True
                if cls_name == gt["cls"]:
                    detected_gt[idx] = True
                    print(f"  [PASS] Correctly detected {gt['desc']} ({cls_name}) at [{x1}, {y1}, {x2}, {y2}] with conf {conf:.2f} (dist: {dist:.1f}px)")
                else:
                    print(f"  [FAIL] Detected flag at {gt['desc']} position but wrong class: got '{cls_name}', expected '{gt['cls']}'")
                break
        
        if not matched:
            false_positives += 1
            print(f"  [FALSE POSITIVE] Detected '{cls_name}' at [{x1}, {y1}, {x2}, {y2}] with conf {conf:.2f} (not close to any GT flag)")
            
    # Check if any GT flag was missed
    missed = 0
    for idx, gt in enumerate(gt_list):
        if not detected_gt[idx]:
            missed += 1
            print(f"  [MISS] Missed {gt['desc']} ({gt['cls']}) at center ({gt['cx']}, {gt['cy']})")
            
    success = (missed == 0 and false_positives == 0)
    return success, missed, false_positives

def test_on_validate_ai():
    # Find weights
    runs_dir = 'runs/detect'
    train_dirs = glob.glob(os.path.join(runs_dir, 'train*'))
    train_dirs.sort(key=os.path.getmtime, reverse=True)
    best_w = None
    for d in train_dirs:
        p = os.path.join(d, 'weights', 'best.pt')
        if os.path.exists(p):
            best_w = p
            break

    if not best_w:
        print("No weights found!")
        return False
        
    print(f"Loading weights: {best_w}")
    model = YOLO(best_w)
    out_dir = 'validate_ai_results_3840'
    os.makedirs(out_dir, exist_ok=True)
    
    validation_passed = True
    total_missed = 0
    total_false_positives = 0
    
    for img_path in sorted(glob.glob('validate_ai/*.jpg')):
        filename = os.path.basename(img_path)
        print(f"\nProcessing {filename} at imgsz=3840...")
        results = model.predict(img_path, conf=0.10, imgsz=3840, agnostic_nms=True)
        img = cv2.imread(img_path)
        
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]
                detections.append((x1, y1, x2, y2, conf, class_name))
                
                # Draw box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                print(f"  Raw Detection: {class_name} ({conf:.2f}) at [{x1}, {y1}, {x2}, {y2}]")
                
        cv2.imwrite(os.path.join(out_dir, f"highres_{filename}"), img)
        
        success, missed, fp = check_detections(filename, detections)
        if filename in GT_FLAGS:
            total_missed += missed
            total_false_positives += fp
            if not success:
                validation_passed = False
                
    print("\n=== Validation Check Summary ===")
    if validation_passed:
        print("ALL CRITICAL VAL FLAGS DETECTED CORRECTLY WITH ZERO FALSE POSITIVES IN 15.JPG!")
        return True
    else:
        print(f"Validation FAILED: Missed {total_missed} flags, Got {total_false_positives} false positives in 15.jpg.")
        return False

if __name__ == '__main__':
    test_on_validate_ai()
