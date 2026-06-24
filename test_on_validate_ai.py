import os
import glob
import cv2
from ultralytics import YOLO

def find_best_weights():
    # Search for runs/detect/train/weights/best.pt or other run folders
    runs_dir = 'runs/detect'
    if not os.path.exists(runs_dir):
        return None
        
    train_dirs = glob.glob(os.path.join(runs_dir, 'train*'))
    if not train_dirs:
        return None
        
    # Sort by modification time to get the latest run
    train_dirs.sort(key=os.path.getmtime, reverse=True)
    
    for d in train_dirs:
        weights_path = os.path.join(d, 'weights', 'best.pt')
        if os.path.exists(weights_path):
            return weights_path
            
    return None

def test_model():
    print("=== Testing YOLOv8 Model on validate_ai/ Images ===")
    
    weights_path = find_best_weights()
    if not weights_path:
        print("Error: Trained model weights ('best.pt') not found under 'runs/detect/train*/weights/'.")
        print("Please run train_flag_yolo.py to train the model first.")
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
        
    output_dir = 'validate_ai_results'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output annotated images will be saved to: {output_dir}\n")
    
    for img_path in sorted(img_paths):
        filename = os.path.basename(img_path)
        print(f"Processing {filename}...")
        
        # Run inference at higher resolution for small objects
        results = model.predict(img_path, conf=0.10, imgsz=1280, agnostic_nms=True)
        
        # Load original image
        img = cv2.imread(img_path)
        
        # Draw predictions
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]
                
                # Draw box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Draw label
                label = f"{class_name} {conf:.2f}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                print(f"  Detected: {class_name} with confidence {conf:.2f} at [{x1}, {y1}, {x2}, {y2}]")
                
        # Save output image
        out_path = os.path.join(output_dir, f"detected_{filename}")
        cv2.imwrite(out_path, img)
        print(f"  Saved detection image to: {out_path}")
        
    print("\nValidation complete! Check the 'validate_ai_results' folder for annotated images.")

if __name__ == '__main__':
    test_model()
