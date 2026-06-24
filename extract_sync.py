import os
import cv2
import numpy as np
from ultralytics import YOLO

def main():
    video_path = r"C:\Users\mido\Documents\antigravity\focused-babbage\GX014604.MP4"
    model_path = r"C:\Users\mido\Documents\antigravity\focused-babbage\yolo26s_flag_best.pt"
    output_dir = r"C:\Users\mido\Documents\antigravity\focused-babbage\new_negatives"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading model...")
    model = YOLO(model_path)
    
    print("Opening video...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    start_frame = 4700
    end_frame = 4900
    print(f"Seeking to frame {start_frame}...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_idx = start_frame
    saved_count = 0
    
    while frame_idx <= end_frame:
        success, frame = cap.read()
        if not success:
            break
            
        results = model.predict(frame, conf=0.15, imgsz=640, verbose=False)
        boxes = results[0].boxes
        
        if len(boxes) > 0:
            print(f"Frame {frame_idx:05d}: Found {len(boxes)} detections")
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
                    out_path = os.path.join(output_dir, f"neg_{frame_idx}_{idx}.jpg")
                    cv2.imwrite(out_path, crop)
                    saved_count += 1
                    print(f"  -> Saved crop to {out_path} (conf={conf:.2f}, bbox_sat={mean_sat:.1f})")
                else:
                    print(f"  -> Ignored (conf={conf:.2f}, bbox_sat={mean_sat:.1f})")
                    
        frame_idx += 1
        
    cap.release()
    print(f"\nDone. Saved {saved_count} negative backgrounds.")

if __name__ == '__main__':
    main()
