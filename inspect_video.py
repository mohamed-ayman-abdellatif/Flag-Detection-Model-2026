import cv2
import os

def inspect(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
        
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Failed to open video: {path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"Video: {path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS:        {fps:.2f}")
    print(f"  Frames:     {frame_count}")
    print(f"  Duration:   {duration:.2f} seconds")
    cap.release()

if __name__ == '__main__':
    inspect(r'C:\Users\mido\Documents\antigravity\focused-babbage\test_flight\GL014208.LRV')
    inspect(r'C:\Users\mido\Documents\antigravity\focused-babbage\test_flight\GX014209.MP4')
