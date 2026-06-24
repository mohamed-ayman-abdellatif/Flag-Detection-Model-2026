import os
import cv2
import numpy as np
from ultralytics import YOLO

def extract_from_segment(video_path, model, start_frame, end_frame, step, output_dir):
    print(f"\nExtracting from {os.path.basename(video_path)}: frames {start_frame} to {end_frame} (step {step})")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return 0
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    saved_count = 0
    
    while frame_idx <= end_frame:
        success, frame = cap.read()
        if not success:
            break
            
        if (frame_idx - start_frame) % step == 0:
            results = model.predict(frame, conf=0.15, imgsz=640, verbose=False)
            boxes = results[0].boxes
            
            if len(boxes) > 0:
                for idx, box in enumerate(boxes):
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy
                    
                    x1_c = max(0, min(x1, width - 1))
                    y1_c = max(0, min(y1, height - 1))
                    x2_c = max(0, min(x2, width - 1))
                    y2_c = max(0, min(y2, height - 1))
                    
                    if (x2_c - x1_c) <= 0 or (y2_c - y1_c) <= 0:
                        continue
                        
                    bbox_crop = frame[y1_c:y2_c, x1_c:x2_c]
                    hsv_box = cv2.cvtColor(bbox_crop, cv2.COLOR_BGR2HSV)
                    mean_sat = np.mean(hsv_box[:, :, 1])
                    
                    cx = (x1_c + x2_c) // 2
                    cy = (y1_c + y2_c) // 2
                    
                    half_sz = 320
                    cx1 = max(0, min(cx - half_sz, width - 640))
                    cy1 = max(0, min(cy - half_sz, height - 640))
                    cx2 = cx1 + 640
                    cy2 = cy1 + 640
                    
                    crop = frame[cy1:cy2, cx1:cx2]
                    
                    if mean_sat < 80.0:
                        # Include video name in filename to avoid overwrites
                        vname = os.path.splitext(os.path.basename(video_path))[0]
                        out_path = os.path.join(output_dir, f"neg_{vname}_{frame_idx}_{idx}.jpg")
                        cv2.imwrite(out_path, crop)
                        saved_count += 1
                        
        frame_idx += 1
        
    cap.release()
    print(f"Saved {saved_count} negative backgrounds from this segment.")
    return saved_count

def main():
    video1 = r"C:\Users\mido\Documents\antigravity\focused-babbage\GX014604.MP4"
    video2 = r"C:\Users\mido\Documents\antigravity\focused-babbage\test_flight\GX014209.MP4"
    model_path = r"C:\Users\mido\Documents\antigravity\focused-babbage\yolo26s_flag_best.pt"
    output_dir = r"C:\Users\mido\Documents\antigravity\focused-babbage\new_negatives"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean previous temp files in new_negatives to start clean
    for f in os.listdir(output_dir):
        if f.endswith(".jpg"):
            try: os.remove(os.path.join(output_dir, f))
            except: pass
            
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    model = YOLO(model_path)
    total_saved = 0
    
    # Video 1 Segments
    if os.path.exists(video1):
        # Scan segment 1: 4700 to 4900 (dense curbs/runways)
        total_saved += extract_from_segment(video1, model, 4700, 4900, 2, output_dir)
        # Scan segment 2: 5600 to 5750 (rocky ground phase)
        total_saved += extract_from_segment(video1, model, 5600, 5750, 2, output_dir)
        
    # Video 2 Segments
    if os.path.exists(video2):
        # Scan segment 1: 2000 to 2300 (takeoff/runway desaturated detections)
        total_saved += extract_from_segment(video2, model, 2000, 2300, 2, output_dir)
        
    print(f"\nTotal custom desaturated negative backgrounds extracted: {total_saved}")

if __name__ == '__main__':
    main()
