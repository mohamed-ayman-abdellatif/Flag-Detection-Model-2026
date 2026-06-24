import os
import cv2
from ultralytics import YOLO

def main():
    video_path = r"C:\Users\mido\Documents\antigravity\focused-babbage\GX014604.MP4"
    model_path = r"C:\Users\mido\Documents\antigravity\focused-babbage\yolo26s_flag_best.pt"
    
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        return
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    print("Loading new model...")
    model = YOLO(model_path)
    
    print("Opening video...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
        
    # We test the exact frames where high confidence false positives occurred in the previous model
    test_frames = [4770, 4800, 4830, 4860, 4890]
    print(f"\nEvaluating new model on known false positive frames: {test_frames}")
    
    for f_idx in test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        success, frame = cap.read()
        if not success:
            print(f"  Frame {f_idx:05d}: Failed to read frame.")
            continue
            
        results = model.predict(frame, conf=0.15, imgsz=640, verbose=False)
        boxes = results[0].boxes
        
        if len(boxes) > 0:
            print(f"  Frame {f_idx:05d}: WARNING! Detections still present:")
            for box in boxes:
                print(f"    -> conf={float(box.conf[0]):.2f}, bbox={box.xyxy[0].cpu().numpy().tolist()}")
        else:
            print(f"  Frame {f_idx:05d}: CLEAN! No detections (False positive resolved)")
            
    cap.release()

if __name__ == '__main__':
    main()
