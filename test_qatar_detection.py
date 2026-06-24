import os
import random
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Reuse background and overlay logic from generate_1class_dataset to create a test image
from generate_1class_dataset import extract_backgrounds, overlay_flag

def main():
    weights_path = r"C:\Users\mido\Documents\antigravity\focused-babbage\yolo26s_flag_best.pt"
    qa_template_path = r"C:\Users\mido\Documents\antigravity\focused-babbage\reference\country\qa.png"
    
    if not os.path.exists(weights_path):
        print(f"Error: Weights not found at {weights_path}")
        return
    if not os.path.exists(qa_template_path):
        print(f"Error: Qatar flag template not found at {qa_template_path}")
        return
        
    print("Loading model and backgrounds...")
    model = YOLO(weights_path)
    bgs = extract_backgrounds()
    
    if not bgs:
        print("Error: No backgrounds extracted.")
        return
        
    # Load Qatar flag template
    with Image.open(qa_template_path) as img:
        rgba = img.convert('RGBA')
        bw = 300
        bh = int((bw / rgba.size[0]) * rgba.size[1])
        qa_flag = rgba.resize((bw, bh), Image.Resampling.LANCZOS)
        
    print("\n--- Running simulated Qatar Flag detection sweep ---")
    
    success_count = 0
    test_runs = 10
    
    for i in range(test_runs):
        bg = random.choice(bgs).copy()
        test_img, label = overlay_flag(bg, qa_flag)
        
        # Save temp image for inference
        temp_img_path = "temp_qatar_test.jpg"
        test_img.save(temp_img_path)
        
        # Run inference
        results = model.predict(temp_img_path, conf=0.10, imgsz=640, verbose=False)
        
        # Parse results
        detected = False
        for r in results:
            if len(r.boxes) > 0:
                detected = True
                conf = float(r.boxes[0].conf[0])
                print(f"  Run {i+1:02d}: Flag DETECTED with confidence {conf:.2f}")
                success_count += 1
                break
                
        if not detected:
            print(f"  Run {i+1:02d}: Flag MISSED")
            
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
            
    success_rate = (success_count / test_runs) * 100
    print("\n" + "="*40)
    print(f"Qatar Flag Detection Success Rate: {success_rate:.1f}% ({success_count}/{test_runs})")
    print("="*40)

if __name__ == '__main__':
    main()
