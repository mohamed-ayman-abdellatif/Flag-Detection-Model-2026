import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

def main():
    video_path = r"C:\Users\mido\Documents\antigravity\focused-babbage\GX014604.MP4"
    model_path = r"C:\Users\mido\Documents\antigravity\focused-babbage\yolo26s_flag_best.pt"
    output_dir = r"C:\Users\mido\Documents\antigravity\focused-babbage\new_negatives"
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    print("Loading model...")
    model = YOLO(model_path)
    
    print("Opening video...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolution: {width}x{height}, Total Frames: {total_frames}")
    
    frame_idx = 0
    step = 5  # Scan every 5th frame to capture variations but keep it fast
    saved_count = 0
    
    print("\n--- Starting Active Negative Extraction (Revised Bounding Box Saturation Filter) ---")
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        if frame_idx % step != 0:
            frame_idx += 1
            continue
            
        # Run prediction
        results = model.predict(frame, conf=0.15, imgsz=640, verbose=False)
        boxes = results[0].boxes
        
        if len(boxes) > 0:
            for idx, box in enumerate(boxes):
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                
                # Bounding box dimensions
                x1, y1, x2, y2 = xyxy
                
                # Clip coordinates to frame boundaries
                x1_c = max(0, min(x1, width - 1))
                y1_c = max(0, min(y1, height - 1))
                x2_c = max(0, min(x2, width - 1))
                y2_c = max(0, min(y2, height - 1))
                
                if (x2_c - x1_c) <= 0 or (y2_c - y1_c) <= 0:
                    continue
                    
                bbox_crop = frame[y1_c:y2_c, x1_c:x2_c]
                
                # Check color saturation of the detected bbox region itself
                hsv_box = cv2.cvtColor(bbox_crop, cv2.COLOR_BGR2HSV)
                mean_sat = np.mean(hsv_box[:, :, 1])
                
                # Bounding box coordinates for the 640x640 patch center
                cx = (x1_c + x2_c) // 2
                cy = (y1_c + y2_c) // 2
                
                # Crop a 640x640 region centered around the detection
                half_sz = 320
                cx1 = max(0, min(cx - half_sz, width - 640))
                cy1 = max(0, min(cy - half_sz, height - 640))
                cx2 = cx1 + 640
                cy2 = cy1 + 640
                
                crop = frame[cy1:cy2, cx1:cx2]
                
                # Concrete barriers, runways, sand, rocks are desaturated (< 80)
                # Colored flags are very saturated (> 80)
                # We save desaturated candidates as false positive background negatives
                if mean_sat < 80.0:
                    out_path = os.path.join(output_dir, f"neg_{frame_idx}_{idx}.jpg")
                    cv2.imwrite(out_path, crop)
                    saved_count += 1
                    print(f"  Frame {frame_idx:04d}: Saved FP background (conf={conf:.2f}, bbox_sat={mean_sat:.1f})")
                    sys.stdout.flush()
                else:
                    # Likely a real flag or highly saturated object
                    print(f"  Frame {frame_idx:04d}: Ignored candidate (conf={conf:.2f}, bbox_sat={mean_sat:.1f} >= 80)")
                    sys.stdout.flush()
                    
        frame_idx += 1
        if frame_idx % 200 == 0:
            print(f"Progress: {frame_idx}/{total_frames} frames processed...", end='\r')
            sys.stdout.flush()
            
    cap.release()
    print(f"\nActive negative extraction complete! Saved {saved_count} negative backgrounds to {output_dir}")
    sys.stdout.flush()

if __name__ == '__main__':
    main()
