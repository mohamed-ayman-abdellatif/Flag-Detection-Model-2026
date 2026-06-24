import os
import glob
import random
import csv
from PIL import Image
from ultralytics import YOLO

# Import backgrounds and overlay logic from generate_1class_dataset
from generate_1class_dataset import extract_backgrounds, overlay_flag

def main():
    weights_path = r"C:\Users\mido\Documents\antigravity\focused-babbage\yolo26s_flag_best.pt"
    ref_country_dir = r"C:\Users\mido\Documents\antigravity\focused-babbage\reference\country"
    ref_inst_dir = r"C:\Users\mido\Documents\antigravity\focused-babbage\reference\institution"
    csv_output_path = r"C:\Users\mido\Documents\antigravity\focused-babbage\validation_sweep_results.csv"
    
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
    
    total_classes = len(template_paths)
    results_list = []
    
    print("\n--- Starting validation sweep (10 tests per flag class, completely in-memory) ---")
    
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
        test_images = []
        
        # Prepare 10 test images in memory
        for run in range(test_runs):
            bg = random.choice(bgs).copy()
            test_img, label = overlay_flag(bg, flag_img)
            test_images.append(test_img)
            
        # Run prediction in batch (passing PIL images directly to model.predict)
        # We use conf=0.10 to match the detection threshold used in the pipeline
        results = model.predict(test_images, conf=0.10, imgsz=640, verbose=False)
        
        # Evaluate results
        for r in results:
            if len(r.boxes) > 0:
                success_count += 1
                
        success_rate = (success_count / test_runs) * 100
        status = "PASS" if success_rate >= 90.0 else "FAIL"
        
        results_list.append({
            "Class Name": name,
            "Status": status,
            "Success Rate": f"{success_rate:.1f}%",
            "Detections": f"{success_count}/{test_runs}"
        })
        
        print(f"  [{idx+1:03d}/{total_classes:03d}] {name}: {success_rate:.1f}% ({success_count}/{test_runs}) - {status}")

    # Write to CSV
    with open(csv_output_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["Class Name", "Status", "Success Rate", "Detections"])
        writer.writeheader()
        writer.writerows(results_list)
        
    print(f"\nSaved exact success rates to {csv_output_path}")

if __name__ == '__main__':
    main()
