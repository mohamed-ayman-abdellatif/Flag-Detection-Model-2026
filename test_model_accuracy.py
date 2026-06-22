import os
import glob
import cv2
from ultralytics import YOLO

# Ground truth locations of the flags in 15.jpg (Germany, Russia, France)
GT_FLAGS = [
    {"desc": "Germany", "cx": 1087, "cy": 142},
    {"desc": "Russia", "cx": 1194, "cy": 2951},
    {"desc": "France", "cx": 2141, "cy": 1392}
]

def main():
    weights_path = r"C:\Users\mido\Documents\antigravity\focused-babbage\yolo26s_flag_best.pt"
    validate_dir = r"C:\Users\mido\Documents\antigravity\focused-babbage\validate_ai"
    output_dir = r"C:\Users\mido\Documents\antigravity\focused-babbage\validate_ai_results_3840"
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(weights_path):
        print(f"Error: Weights not found at {weights_path}")
        return
        
    print(f"Loading YOLO26 model weights: {weights_path}")
    model = YOLO(weights_path)
    
    img_paths = sorted(glob.glob(os.path.join(validate_dir, "*.jpg")))
    if not img_paths:
        print(f"Error: No images found in {validate_dir}")
        return
        
    print(f"Found {len(img_paths)} validation images. Starting inference at imgsz=3840...")
    
    # Trackers for 15.jpg accuracy
    gt_detected = [False] * len(GT_FLAGS)
    false_positives_15 = 0
    
    for img_path in img_paths:
        filename = os.path.basename(img_path)
        print(f"\nProcessing {filename}...")
        
        # Run prediction at 640px resolution (matching training imgsz)
        results = model.predict(img_path, conf=0.10, imgsz=640, agnostic_nms=True, verbose=False)
        
        img = cv2.imread(img_path)
        
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]
                detections.append((x1, y1, x2, y2, conf, class_name))
                
                # Draw green bounding box and confidence label
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
        # Save annotated image
        out_path = os.path.join(output_dir, f"highres_{filename}")
        cv2.imwrite(out_path, img)
        print(f"  Detected {len(detections)} flags. Saved annotated image to {out_path}")
        
        # Evaluate accuracy specifically for 15.jpg
        if filename == "15.jpg":
            print("  Evaluating detections on 15.jpg:")
            for det in detections:
                x1, y1, x2, y2, conf, cls_name = det
                dcx = (x1 + x2) / 2
                dcy = (y1 + y2) / 2
                
                matched = False
                for idx, gt in enumerate(GT_FLAGS):
                    dist = ((dcx - gt["cx"])**2 + (dcy - gt["cy"])**2)**0.5
                    if dist < 120:  # 120 pixels radius (increased slightly for 4K margin)
                        gt_detected[idx] = True
                        matched = True
                        print(f"    [MATCH] Found {gt['desc']} at [{x1}, {y1}, {x2}, {y2}] (dist: {dist:.1f}px, conf: {conf:.2f})")
                        break
                        
                if not matched:
                    false_positives_15 += 1
                    print(f"    [EXTRA] Extra detection at [{x1}, {y1}, {x2}, {y2}] (conf: {conf:.2f})")
                    
            for idx, gt in enumerate(GT_FLAGS):
                if not gt_detected[idx]:
                    print(f"    [MISS] Missed {gt['desc']} at ground-truth ({gt['cx']}, {gt['cy']})")
                    
    # Calculate accuracy metrics
    detected_count = sum(gt_detected)
    accuracy_percentage = (detected_count / len(GT_FLAGS)) * 100
    
    print("\n" + "="*50)
    print("=== SUMMARY METRICS FOR 15.JPG ===")
    print(f"Accuracy (Ground Truth Detection Rate): {accuracy_percentage:.1f}% ({detected_count}/{len(GT_FLAGS)} flags)")
    print(f"Missed Flags: {len(GT_FLAGS) - detected_count}")
    print(f"Extra Detections (False Positives): {false_positives_15}")
    print("="*50)

if __name__ == "__main__":
    main()
