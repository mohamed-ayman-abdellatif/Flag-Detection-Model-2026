import os
import sys
import shutil
import random
import subprocess
from PIL import Image

def clean_dir(path):
    import time
    for _ in range(5):
        try:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)
            return
        except Exception:
            time.sleep(0.5)
    # Fallback to direct call which throws on failure
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def main():
    base_dir = r"C:\Users\mido\Documents\antigravity\focused-babbage"
    src_dir = os.path.join(base_dir, "data set-20260622T155136Z-3-001", "data set", "dataset")
    synth_dir = os.path.join(base_dir, "synthetic_dataset")
    dst_dir = os.path.join(base_dir, "kaggle_dataset_resized")

    # 1. Run generate_1class_dataset.py to generate new synthetic data
    print("=== Cleaning old synthetic dataset folder ===")
    clean_dir(synth_dir)
    print("=== Generating new synthetic dataset ===")
    gen_script = os.path.join(base_dir, "generate_1class_dataset.py")
    subprocess.check_call([sys.executable, gen_script])

    # 2. Reset the target merged dataset directory
    print("\n=== Initializing merged dataset folder ===")
    for split in ['train', 'val']:
        clean_dir(os.path.join(dst_dir, "images", split))
        clean_dir(os.path.join(dst_dir, "labels", split))

    # 3. Process and merge original user dataset (with 80/20 train/val split & resize to 640)
    print("\n=== Merging and resizing original dataset images ===")
    src_images = os.path.join(src_dir, "images")
    src_labels = os.path.join(src_dir, "labels")

    image_files = sorted([f for f in os.listdir(src_images) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    # Set seed for deterministic split
    random.seed(42)
    
    count = 0
    total = len(image_files)
    
    for img_name in image_files:
        # Determine train/val split
        split = "train" if random.random() < 0.80 else "val"
        
        src_img_path = os.path.join(src_images, img_name)
        dst_img_path = os.path.join(dst_dir, "images", split, f"original_{img_name}")
        
        # Resize image
        try:
            with Image.open(src_img_path) as img:
                img.thumbnail((640, 640))
                img.save(dst_img_path, "JPEG", quality=85)
        except Exception as e:
            print(f"Failed to resize {img_name}: {e}")
            continue

        # Copy/Create label file
        lbl_name = os.path.splitext(img_name)[0] + ".txt"
        src_lbl_path = os.path.join(src_labels, lbl_name)
        dst_lbl_path = os.path.join(dst_dir, "labels", split, f"original_{lbl_name}")
        
        if os.path.exists(src_lbl_path):
            shutil.copy2(src_lbl_path, dst_lbl_path)
        else:
            # Write empty label file if none exists (negative background sample)
            with open(dst_lbl_path, 'w') as lf:
                pass
                
        count += 1
        if count % 100 == 0 or count == total:
            print(f"  Processed {count}/{total} original images...")

    # 4. Copy new synthetic dataset into splits
    print("\n=== Merging new synthetic dataset ===")
    for split in ['train', 'val']:
        synth_img_dir = os.path.join(synth_dir, "images", split)
        synth_lbl_dir = os.path.join(synth_dir, "labels", split)
        
        # Copy images
        s_imgs = os.listdir(synth_img_dir)
        for s_img in s_imgs:
            shutil.copy2(os.path.join(synth_img_dir, s_img), os.path.join(dst_dir, "images", split, s_img))
            
        # Copy labels
        s_lbls = os.listdir(synth_lbl_dir)
        for s_lbl in s_lbls:
            shutil.copy2(os.path.join(synth_lbl_dir, s_lbl), os.path.join(dst_dir, "labels", split, s_lbl))
            
        print(f"  Copied {len(s_imgs)} synthetic images to {split} split.")

    # 4b. Copy new full-frame negatives into splits with empty labels
    print("\n=== Merging new full-frame negatives ===")
    full_neg_dir = os.path.join(base_dir, "full_frame_negatives")
    if os.path.exists(full_neg_dir):
        neg_files = [f for f in os.listdir(full_neg_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        train_count = 0
        val_count = 0
        for f in neg_files:
            # 80/20 split
            split = "train" if random.random() < 0.80 else "val"
            
            # Copy image
            src_img = os.path.join(full_neg_dir, f)
            dst_img = os.path.join(dst_dir, "images", split, f)
            shutil.copy2(src_img, dst_img)
            
            # Write empty label file
            lbl_name = os.path.splitext(f)[0] + ".txt"
            dst_lbl = os.path.join(dst_dir, "labels", split, lbl_name)
            with open(dst_lbl, 'w') as lf:
                pass
                
            if split == "train":
                train_count += 1
            else:
                val_count += 1
                
        print(f"  Copied {train_count} full-frame negatives to train split, and {val_count} to val split.")

    # 4c. Copy target image and write labels (30x oversampled in train split)
    print("\n=== Merging and oversampling target image ===")
    target_img_src = r"C:\Users\mido\.gemini\antigravity\brain\5fbcdffe-e6b3-4e67-8084-371c60076362\media__1782355964878.jpg"
    target_labels = [
        "0 0.145854 0.176285 0.008857 0.017780",
        "0 0.172876 0.491302 0.011572 0.010787",
        "0 0.425854 0.495428 0.014521 0.018374",
        "0 0.458804 0.040490 0.008916 0.022238",
        "0 0.620708 0.333776 0.009658 0.015490"
    ]
    if os.path.exists(target_img_src):
        for idx in range(30):
            target_img_name = f"target_frame_{idx:02d}.jpg"
            target_lbl_name = f"target_frame_{idx:02d}.txt"
            
            # Copy to train images
            shutil.copy2(target_img_src, os.path.join(dst_dir, "images", "train", target_img_name))
            
            # Write to train labels
            target_lbl_dst = os.path.join(dst_dir, "labels", "train", target_lbl_name)
            with open(target_lbl_dst, 'w') as lf:
                for lbl in target_labels:
                    lf.write(lbl + "\n")
        print("  Copied 30 oversampled copies of target image to train split.")
    else:
        print("  WARNING: Target image not found at target_img_src!")

    # 4d. Copy pre-trained model weights into the dataset folder
    print("\n=== Copying pre-trained model weights ===")
    best_weights_src = os.path.join(base_dir, "yolo26s_flag_best.pt")
    if os.path.exists(best_weights_src):
        shutil.copy2(best_weights_src, os.path.join(dst_dir, "yolo26s_flag_best.pt"))
        print("  Copied yolo26s_flag_best.pt into merged dataset folder.")

    # 5. Generate final data.yaml config
    yaml_content = f"""path: {dst_dir.replace('\\', '/')}
train: images/train
val: images/val

# Number of classes
nc: 1

# Class names
names: ['flag']
"""
    with open(os.path.join(dst_dir, "data.yaml"), "w") as f:
        f.write(yaml_content)
        
    print(f"\n=== Merging complete! Merged dataset is ready under C:\\Users\\mido\\Documents\\antigravity\\focused-babbage\\kaggle_dataset_resized ===")

if __name__ == '__main__':
    main()
