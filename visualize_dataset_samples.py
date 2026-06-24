import os
import cv2
import glob
import random
import numpy as np

CLASS_NAMES = {
    0: 'Egypt',
    1: 'France',
    2: 'Germany',
    3: 'Republic of Korea',
    4: 'NATO',
    5: 'Japan',
    6: 'Italy',
    7: 'United States',
    8: 'Palestine',
    9: 'Canada',
    10: 'United Kingdom',
    11: 'Russia'
}

CLASS_COLORS = {
    0: (0, 0, 255),       # Red
    1: (255, 0, 0),       # Blue
    2: (0, 255, 255),     # Yellow
    3: (0, 255, 0),       # Green
    4: (255, 0, 255),     # Magenta
    5: (255, 128, 0),     # Orange
    6: (0, 128, 255),     # Azure
    7: (128, 0, 255),     # Purple
    8: (0, 255, 128),     # Mint
    9: (128, 255, 0),     # Lime
    10: (0, 128, 128),    # Teal
    11: (255, 255, 255)   # White
}

def draw_yolo_labels(img_path, label_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
        
    h, w, _ = img.shape
    if not os.path.exists(label_path):
        print(f"Label not found: {label_path}")
        return img
        
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        class_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:])
        
        # Convert to pixel coordinates
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        
        color = CLASS_COLORS.get(class_id, (0, 255, 0))
        name = CLASS_NAMES.get(class_id, f'Class_{class_id}')
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Label text
        label_text = f"{name}"
        cv2.putText(img, label_text, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
    return img

def visualize_samples(num_samples=5):
    dataset_dir = r'C:\Users\mido\Documents\antigravity\focused-babbage\synthetic_dataset'
    images_dir = os.path.join(dataset_dir, 'images', 'train')
    labels_dir = os.path.join(dataset_dir, 'labels', 'train')
    
    img_paths = glob.glob(os.path.join(images_dir, '*.jpg'))
    if not img_paths:
        print("No synthetic images found! Run generate_synthetic_dataset.py first.")
        return
        
    sampled_paths = random.sample(img_paths, min(num_samples, len(img_paths)))
    
    # Create verification output directory
    out_dir = r'C:\Users\mido\Documents\antigravity\focused-babbage\sample_verification'
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nSaving verified sample images to: {out_dir}")
    
    for idx, img_path in enumerate(sampled_paths):
        filename = os.path.basename(img_path)
        lbl_filename = filename.replace('.jpg', '.txt')
        lbl_path = os.path.join(labels_dir, lbl_filename)
        
        annotated_img = draw_yolo_labels(img_path, lbl_path)
        if annotated_img is not None:
            out_path = os.path.join(out_dir, f"verified_{filename}")
            cv2.imwrite(out_path, annotated_img)
            print(f"  Saved verified sample: {os.path.basename(out_path)}")
            
            # Also copy one sample to the brain artifacts folder for easy viewer access
            if idx == 0:
                brain_out_path = r'C:\Users\mido\.gemini\antigravity\brain\7c583763-3713-4256-b5ad-dfa91099fc03\sample_verification.jpg'
                cv2.imwrite(brain_out_path, annotated_img)
                print(f"  Saved display sample to artifacts: {brain_out_path}")

if __name__ == '__main__':
    visualize_samples()
