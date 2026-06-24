import os
import glob
from ultralytics import YOLO

def main():
    model_path = r"C:\Users\mido\Documents\antigravity\focused-babbage\yolo26s_flag_best.pt"
    frames_dir = r"c:\Users\mido\Downloads\frames-20260515T124041Z-3-001\frames"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    model = YOLO(model_path)
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    print(f"Found {len(frame_paths)} frames. Running inference...")
    
    detections = []
    # Process in batches to be fast
    batch_size = 32
    for i in range(0, len(frame_paths), batch_size):
        batch_paths = frame_paths[i:i+batch_size]
        results = model.predict(batch_paths, conf=0.10, imgsz=640, verbose=False)
        for path, r in zip(batch_paths, results):
            fname = os.path.basename(path)
            if len(r.boxes) > 0:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()
                    detections.append((fname, conf, xyxy))
                    print(f"Detection in {fname}: conf={conf:.2f}, bbox={xyxy}")
                    
    print(f"\nTotal detections: {len(detections)}")

if __name__ == '__main__':
    main()
