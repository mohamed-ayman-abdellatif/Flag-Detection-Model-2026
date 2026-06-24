import os
import cv2
from ultralytics import YOLO

def scan_video(video_path, model):
    print(f"\nScanning: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    step = 30  # Scan every 30th frame for quick search (1 frame per second)
    det_count = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        if frame_idx % step == 0:
            results = model.predict(frame, conf=0.15, imgsz=640, verbose=False)
            boxes = results[0].boxes
            if len(boxes) > 0:
                det_count += len(boxes)
                for idx, box in enumerate(boxes):
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    # Compute box saturation
                    x1, y1, x2, y2 = xyxy
                    crop = frame[y1:y2, x1:x2]
                    mean_sat = 0
                    if crop.size > 0:
                        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                        mean_sat = float(hsv[:, :, 1].mean())
                    print(f"  Frame {frame_idx:05d}: conf={conf:.2f}, bbox={xyxy.tolist()}, sat={mean_sat:.1f}")
                    
        frame_idx += 1
    cap.release()
    print(f"Total detections in {os.path.basename(video_path)}: {det_count}")

def main():
    model_path = r"C:\Users\mido\Documents\antigravity\focused-babbage\yolo26s_flag_best.pt"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    model = YOLO(model_path)
    
    videos = [
        r"C:\Users\mido\Documents\antigravity\focused-babbage\GX014604.MP4",
        r"C:\Users\mido\Documents\antigravity\focused-babbage\test_flight\GX014209.MP4"
    ]
    
    for v in videos:
        if os.path.exists(v):
            scan_video(v, model)
        else:
            print(f"Video {v} not found.")

if __name__ == '__main__':
    main()
