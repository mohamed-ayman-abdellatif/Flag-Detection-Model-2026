import cv2
import numpy as np
import os
import glob
import csv

def detect_flag_in_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Red color range
    mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([12, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([165, 70, 50]), np.array([180, 255, 255]))
    red_mask = mask1 | mask2
    
    # Find contours of red
    cnts, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 5 or area > 2000:
            continue
            
        x, y, w, h = cv2.boundingRect(c)
        # The flag is a rectangle on the ground, the red stripe is the top stripe.
        # Let's check the neighborhood of the red stripe to see if there is white and black below it.
        # We look at a region of height ~ 3.5 * h and width ~ w, starting at y.
        # Since the flag could be rotated, we can check a slightly larger bounding box.
        
        # Let's crop a region of 3 * h height and 1.5 * w width below the red stripe
        margin_y = int(h * 0.5)
        margin_x = int(w * 0.3)
        
        y1 = max(0, y - margin_y)
        y2 = min(img.shape[0], y + h * 4)
        x1 = max(0, x - margin_x)
        x2 = min(img.shape[1], x + w + margin_x)
        
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
            
        # Convert crop to HSV
        crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # In the crop:
        # We expect a red region near the top.
        # We expect a white region in the middle: high value, low saturation.
        # We expect a black region at the bottom: low value.
        
        # Let's compute average intensity/saturation in vertical slices
        # Since the flag can be tilted, let's do a simple color check:
        # Does the crop contain white and black?
        
        # White mask: S < 60, V > 160
        white_mask = cv2.inRange(crop_hsv, np.array([0, 0, 160]), np.array([180, 60, 255]))
        # Black mask: V < 70
        black_mask = cv2.inRange(crop_hsv, np.array([0, 0, 0]), np.array([180, 255, 70]))
        
        num_white = np.sum(white_mask > 0)
        num_black = np.sum(black_mask > 0)
        
        # The red contour area in the crop is:
        num_red = np.sum(red_mask[y1:y2, x1:x2] > 0)
        
        # Let's check if the vertical order is roughly Red -> White -> Black
        # We can find the centroids of the Red, White, and Black pixels in the crop
        red_pts = np.argwhere(red_mask[y1:y2, x1:x2] > 0)
        white_pts = np.argwhere(white_mask > 0)
        black_pts = np.argwhere(black_mask > 0)
        
        if len(red_pts) > 3 and len(white_pts) > 3 and len(black_pts) > 3:
            mean_red_y = np.mean(red_pts[:, 0])
            mean_white_y = np.mean(white_pts[:, 0])
            mean_black_y = np.mean(black_pts[:, 0])
            
            # Since the flag is Red-White-Black, we expect:
            # mean_red_y < mean_white_y < mean_black_y  (top to bottom in image coordinates)
            # OR if the image is upside down (unlikely for drone unless flying backwards/weird gimbal):
            # but usually Red is at the top, Black at the bottom.
            # Let's check if they are ordered vertically.
            
            # Also check if their X positions are close
            mean_red_x = np.mean(red_pts[:, 1])
            mean_white_x = np.mean(white_pts[:, 1])
            mean_black_x = np.mean(black_pts[:, 1])
            
            x_dist = max(mean_red_x, mean_white_x, mean_black_x) - min(mean_red_x, mean_white_x, mean_black_x)
            
            # Bounding box width
            crop_w = x2 - x1
            crop_h = y2 - y1
            
            # R-W-B should be aligned vertically
            if mean_red_y < mean_white_y < mean_black_y and x_dist < crop_w * 0.4:
                # This is a very strong candidate!
                # Let's compute the flag center
                # The flag center is roughly the white stripe center
                cx = x1 + mean_white_x
                cy = y1 + mean_white_y
                candidates.append((cx, cy, area, x, y, w, h))
                
    if not candidates:
        return None
        
    # If multiple candidates, choose the one that has area closest to expected or largest area
    # In frame 700 we saw area=262, let's sort by area descending
    candidates.sort(key=lambda item: item[2], reverse=True)
    return candidates[0] # Returns (cx, cy, area, x, y, w, h)

def detect_all():
    frames_dir = r'c:\Users\mido\Downloads\frames-20260515T124041Z-3-001\frames'
    img_paths = sorted(glob.glob(os.path.join(frames_dir, 'frame_*.jpg')))
    
    results = []
    found_count = 0
    for idx, path in enumerate(img_paths):
        filename = os.path.basename(path)
        res = detect_flag_in_image(path)
        if res:
            cx, cy, area, x, y, w, h = res
            results.append({
                'frame': filename,
                'detected': 1,
                'x': cx,
                'y': cy,
                'area': area,
                'bbox_x': x,
                'bbox_y': y,
                'bbox_w': w,
                'bbox_h': h
            })
            found_count += 1
            if found_count <= 10 or idx % 50 == 0:
                print(f"Found in {filename}: Center=({cx:.1f}, {cy:.1f}), Area={area}")
        else:
            results.append({
                'frame': filename,
                'detected': 0,
                'x': None,
                'y': None,
                'area': None,
                'bbox_x': None,
                'bbox_y': None,
                'bbox_w': None,
                'bbox_h': None
            })
            
    with open('flag_detections.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame', 'detected', 'x', 'y', 'area', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h'])
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nDone! Detected flag in {found_count} / {len(img_paths)} frames. Saved to flag_detections.csv")

if __name__ == '__main__':
    detect_all()
