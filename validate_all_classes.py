import os
import glob
import random
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Import backgrounds and overlay logic from generate_1class_dataset
from generate_1class_dataset import extract_backgrounds, overlay_flag

def main():
    weights_path = r"C:\Users\mido\Documents\antigravity\focused-babbage\yolo26s_flag_best.pt"
    ref_country_dir = r"C:\Users\mido\Documents\antigravity\focused-babbage\reference\country"
    ref_inst_dir = r"C:\Users\mido\Documents\antigravity\focused-babbage\reference\institution"
    
    if not os.path.exists(weights_path):
        print(f"Error: Weights not found at {weights_path}")
        return
        
    print("Loading model and backgrounds...")
    model = YOLO(weights_path)
    bgs = extract_backgrounds()
    
    if not bgs:
        print("Error: No backgrounds extracted.")
        return
        
    # Load all template paths
    template_paths = {}
    for path in glob.glob(os.path.join(ref_country_dir, "*")):
        if path.lower().endswith(".png"):
            name = os.path.splitext(os.path.basename(path))[0]
            template_paths[name] = path
    for path in glob.glob(os.path.join(ref_inst_dir, "*")):
        if path.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
            name = os.path.splitext(os.path.basename(path))[0]
            template_paths[name] = path
            
    print(f"Loaded {len(template_paths)} flag classes.")
    
    temp_dir = "temp_validation_images"
    os.makedirs(temp_dir, exist_ok=True)
    
    failing_classes = {}
    passing_count = 0
    total_classes = len(template_paths)
    
    # Process classes in batches or one by one
    # To keep memory footprint low and print progress, we process class-by-class
    print("\n--- Starting validation sweep (10 tests per flag class) ---")
    
    for idx, (name, path) in enumerate(template_paths.items()):
        try:
            with Image.open(path) as img:
                rgba = img.convert('RGBA')
                bw = 300
                bh = int((bw / rgba.size[0]) * rgba.size[1])
                flag_img = rgba.resize((bw, bh), Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"  [{idx+1}/{total_classes}] Skip {name}: {e}")
            continue
            
        success_count = 0
        test_runs = 10
        temp_paths = []
        
        # Prepare 10 test images
        for run in range(test_runs):
            bg = random.choice(bgs).copy()
            test_img, label = overlay_flag(bg, flag_img)
            
            p = os.path.join(temp_dir, f"temp_{name}_{run}.jpg")
            test_img.save(p)
            temp_paths.append(p)
            
        # Run prediction in batch for speed
        results = model.predict(temp_paths, conf=0.10, imgsz=640, verbose=False)
        
        # Evaluate results
        for r in results:
            if len(r.boxes) > 0:
                success_count += 1
                
        # Clean up temp files
        for p in temp_paths:
            if os.path.exists(p):
                try: os.remove(p)
                except: pass
                
        success_rate = (success_count / test_runs) * 100
        
        if success_rate < 90.0:
            failing_classes[name] = success_rate
            print(f"  [{idx+1:03d}/{total_classes:03d}] {name.upper()}: FAIL - Success Rate {success_rate:.1f}% ({success_count}/{test_runs})")
        else:
            passing_count += 1
            if (idx + 1) % 30 == 0 or idx + 1 == total_classes:
                print(f"  [{idx+1:03d}/{total_classes:03d}] Progress check: {passing_count} classes passing so far...")
                
    # Clean up temp folder
    try: shutil.rmtree(temp_dir)
    except: pass
    
    print("\n" + "="*50)
    print("=== SWEEP SUMMARY ===")
    print(f"Total classes evaluated: {total_classes}")
    print(f"Passing classes (>=90%): {passing_count} ({ (passing_count/total_classes)*100:.1f}%)")
    print(f"Failing classes (<90%): {len(failing_classes)} ({ (len(failing_classes)/total_classes)*100:.1f}%)")
    print("="*50)
    
    if failing_classes:
        print("\nFailing Classes List:")
        for k, v in sorted(failing_classes.items(), key=lambda item: item[1]):
            print(f"  - {k}: {v:.1f}%")
            
        # Save failing classes to a text file for generate_1class_dataset.py to read
        with open("failing_classes.txt", "w") as f:
            for k in failing_classes.keys():
                f.write(k + "\n")
        print("\nSaved failing classes list to failing_classes.txt")
    else:
        if os.path.exists("failing_classes.txt"):
            os.remove("failing_classes.txt")
        print("\nALL CLASSES PASSED WITH >=90% SUCCESS RATE!")

if __name__ == '__main__':
    main()
