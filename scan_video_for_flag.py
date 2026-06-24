import cv2
import numpy as np
import os

def detect_flag_in_frame(img):
    if img is None:
        return None
        
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Red range
    mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([12, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([165, 70, 50]), np.array([180, 255, 255]))
    red_mask = mask1 | mask2
    
    cnts, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        scale = img.shape[1] / 3840.0
        min_area = max(2.0, 5.0 * (scale**2))
        max_area = 2000.0 * (scale**2)
        
        if area < min_area or area > max_area:
            continue
            
        x, y, w, h = cv2.boundingRect(c)
        margin_y = int(h * 0.5)
        margin_x = int(w * 0.3)
        
        y1 = max(0, y - margin_y)
        y2 = min(img.shape[0], y + h * 4)
        x1 = max(0, x - margin_x)
        x2 = min(img.shape[1], x + w + margin_x)
        
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
            
        crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        white_mask = cv2.inRange(crop_hsv, np.array([0, 0, 140]), np.array([180, 60, 255]))
        black_mask = cv2.inRange(crop_hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
        
        red_pts = np.argwhere(red_mask[y1:y2, x1:x2] > 0)
        white_pts = np.argwhere(white_mask > 0)
        black_pts = np.argwhere(black_mask > 0)
        
        if len(red_pts) > 2 and len(white_pts) > 2 and len(black_pts) > 2:
            mean_red_y = np.mean(red_pts[:, 0])
            mean_white_y = np.mean(white_pts[:, 0])
            mean_black_y = np.mean(black_pts[:, 0])
            
            mean_red_x = np.mean(red_pts[:, 1])
            mean_white_x = np.mean(white_pts[:, 1])
            mean_black_x = np.mean(black_pts[:, 1])
            
            x_dist = max(mean_red_x, mean_white_x, mean_black_x) - min(mean_red_x, mean_white_x, mean_black_x)
            crop_w = x2 - x1
            
            if mean_red_y < mean_white_y < mean_black_y and x_dist < crop_w * 0.5:
                cx = x1 + mean_white_x
                cy = y1 + mean_white_y
                candidates.append((cx, cy, area))
                
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[2], reverse=True)
    return candidates[0]

def scan_video_sequential(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
        
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Failed to open video: {path}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nSequential Scanning: {path} ({frame_count} frames)...")
    
    f_idx = 0
    found_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if f_idx % 30 == 0:
            det = detect_flag_in_frame(frame)
            if det:
                time_sec = f_idx / fps
                print(f"  Flag detected at frame {f_idx} ({time_sec:.1f}s): Center=({det[0]:.1f}, {det[1]:.1f}), Area={det[2]:.1f}")
                found_frames.append(f_idx)
                
        f_idx += 1
        
    print(f"Finished scanning {path}. Found flag in {len(found_frames)} frames.")
    cap.release()

if __name__ == '__main__':
    # Scan both files sequentially
    scan_video_sequential(r'C:\Users\mido\Documents\antigravity\focused-babbage\test_flight\GL014208.LRV')
    scan_video_sequential(r'C:\Users\mido\Documents\antigravity\focused-babbage\test_flight\GX014209.MP4')
