import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO

def find_best_weights():
    runs_dir = 'runs/detect'
    if not os.path.exists(runs_dir):
        return None
    train_dirs = glob.glob(os.path.join(runs_dir, 'train*'))
    if not train_dirs:
        return None
    train_dirs.sort(key=os.path.getmtime, reverse=True)
    for d in train_dirs:
        weights_path = os.path.join(d, 'weights', 'best.pt')
        if os.path.exists(weights_path):
            return weights_path
    return None

def nms(boxes, overlap_thresh=0.3):
    if len(boxes) == 0:
        return []
    
    # Convert boxes to float numpy array
    boxes = np.array(boxes, dtype=np.float32)
    
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]
    
    # Compute areas
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)
    
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        # Find the intersection coordinates
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        # Width and height of intersection
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        # Compute the ratio of overlap
        overlap = (w * h) / areas[idxs[:last]]
        
        # Delete all indexes from the index list that have overlap greater than threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
        
    return boxes[pick].tolist()

def run_tiled_inference():
    print("=== Running Tiled Inference on validate_ai/ Images ===")
    
    weights_path = find_best_weights()
    if not weights_path:
        print("Error: Trained model weights ('best.pt') not found.")
        return
        
    print(f"Loading trained weights from: {weights_path}")
    model = YOLO(weights_path)
    
    validate_dir = 'validate_ai'
    if not os.path.exists(validate_dir):
        print(f"Error: Validation directory '{validate_dir}' not found.")
        return
        
    img_paths = glob.glob(os.path.join(validate_dir, '*.jpg'))
    if not img_paths:
        print(f"No JPG images found in '{validate_dir}'.")
        return
        
    output_dir = 'validate_ai_results_tiled'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Annotated images will be saved to: {output_dir}\n")
    
    tile_size = 640
    step_size = 480 # 160 pixels overlap
    
    for img_path in sorted(img_paths):
        filename = os.path.basename(img_path)
        print(f"Processing {filename}...")
        
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        
        all_detections = []
        
        # Generate tiles
        y_steps = list(range(0, h - tile_size, step_size))
        if y_steps[-1] + tile_size < h:
            y_steps.append(h - tile_size)
            
        x_steps = list(range(0, w - tile_size, step_size))
        if x_steps[-1] + tile_size < w:
            x_steps.append(w - tile_size)
            
        tiles = []
        positions = []
        
        for y in y_steps:
            for x in x_steps:
                tile = img[y:y+tile_size, x:x+tile_size]
                tiles.append(tile)
                positions.append((x, y))
                
        # Batch predict tiles (groups of 16)
        batch_size = 16
        for i in range(0, len(tiles), batch_size):
            batch_tiles = tiles[i:i+batch_size]
            batch_positions = positions[i:i+batch_size]
            
            # Predict at same imgsz as training
            results = model.predict(batch_tiles, imgsz=416, conf=0.15, verbose=False)
            
            for r_idx, result in enumerate(results):
                x_offset, y_offset = batch_positions[r_idx]
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # Map back to original image coordinates
                    orig_x1 = x1 + x_offset
                    orig_y1 = y1 + y_offset
                    orig_x2 = x2 + x_offset
                    orig_y2 = y2 + y_offset
                    
                    all_detections.append([orig_x1, orig_y1, orig_x2, orig_y2, conf, cls])
                    
        # Apply class-agnostic NMS to resolve overlapping boxes of different classes
        final_detections = nms(all_detections, overlap_thresh=0.3)
            
        # Draw final detections
        for det in final_detections:
            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cls = int(cls)
            class_name = model.names[cls]
            
            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            # Label
            label = f"{class_name} {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            print(f"  Detected: {class_name} ({conf:.2f}) at [{x1}, {y1}, {x2}, {y2}]")
            
        out_path = os.path.join(output_dir, f"tiled_{filename}")
        cv2.imwrite(out_path, img)
        print(f"  Saved tiled detection image to: {out_path}")
        
    print("\nTiled inference completed! Detections saved.")

if __name__ == '__main__':
    run_tiled_inference()
