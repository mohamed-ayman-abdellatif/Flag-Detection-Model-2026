import subprocess
import os

print("=== Running Validation Scripts ===")

# 1. Run standard test at imgsz=1280
print("\n--- 1. Running standard prediction at imgsz=1280 ---")
subprocess.run(["python", "test_on_validate_ai.py"])

# 2. Run tiled prediction
print("\n--- 2. Running tiled prediction ---")
subprocess.run(["python", "test_on_validate_ai_tiled.py"])

# 3. Run high-resolution prediction at imgsz=3840 (full resolution)
print("\n--- 3. Running prediction at imgsz=3840 ---")
# We will create a quick script for imgsz=3840 or just run it directly
highres_script = """
import os
import glob
import cv2
from ultralytics import YOLO

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

if best_w:
    print(f"Loading weights: {best_w}")
    model = YOLO(best_w)
    out_dir = 'validate_ai_results_3840'
    os.makedirs(out_dir, exist_ok=True)
    
    for img_path in sorted(glob.glob('validate_ai/*.jpg')):
        filename = os.path.basename(img_path)
        print(f"Processing {filename} at imgsz=3840...")
        results = model.predict(img_path, conf=0.10, imgsz=3840, agnostic_nms=True)
        img = cv2.imread(img_path)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                print(f"  Detected: {class_name} ({conf:.2f}) at [{x1}, {y1}, {x2}, {y2}]")
        cv2.imwrite(os.path.join(out_dir, f"highres_{filename}"), img)
else:
    print("No weights found!")
"""

with open("test_on_validate_ai_3840.py", "w") as f:
    f.write(highres_script)

subprocess.run(["python", "test_on_validate_ai_3840.py"])

print("\nAll validation runs completed!")
