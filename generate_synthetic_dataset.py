import os
import cv2
import json
import random
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

# --- 1. LOAD ALL TEMPLATES DYNAMICALLY ---
ref_country_dir = r"C:\Users\mido\Documents\antigravity\focused-babbage\reference\country"
ref_inst_dir = r"C:\Users\mido\Documents\antigravity\focused-babbage\reference\institution"

template_paths = {}

# Load countries
for path in glob.glob(os.path.join(ref_country_dir, "*")):
    if path.lower().endswith(".png"):
        name = os.path.basename(path).replace(".png", "").replace(".PNG", "")
        template_paths[name] = path

# Load institutions
for path in glob.glob(os.path.join(ref_inst_dir, "*")):
    if path.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
        name = os.path.basename(path).split(".")[0]
        template_paths[name] = path

# Sort classes alphabetically to ensure deterministic mapping
sorted_classes = sorted(template_paths.keys())
class_to_id = {name: idx for idx, name in enumerate(sorted_classes)}

print(f"Total dynamic classes loaded: {len(sorted_classes)}")

# --- 2. BACKGROUND Patches Extraction ---
def extract_backgrounds_from_frames():
    frames_dir = r'c:\Users\mido\Downloads\frames-20260515T124041Z-3-001\frames'
    if not os.path.exists(frames_dir):
        print(f"Frames directory not found at: {frames_dir}")
        print("Falling back to creating synthetic ground backgrounds...")
        backgrounds = []
        for _ in range(20):
            bg = Image.new('RGB', (320, 320), (218, 187, 156))
            bg_np = np.array(bg)
            noise = np.random.randint(-15, 15, bg_np.shape, dtype=np.int16)
            bg_np = np.clip(bg_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            backgrounds.append(Image.fromarray(bg_np))
        return backgrounds
        
    print(f"Extracting ground backgrounds from frames in {frames_dir}...")
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    
    indices = np.linspace(0, len(frame_files) - 1, 15, dtype=int)
    sampled_files = [frame_files[idx] for idx in indices]
    
    backgrounds = []
    for fname in sampled_files:
        path = os.path.join(frames_dir, fname)
        try:
            with Image.open(path) as img:
                w, h = img.size
                patch_w, patch_h = 320, 320
                crops = [
                    (100, 100, 100 + patch_w, 100 + patch_h),
                    (w - 100 - patch_w, 100, w - 100, 100 + patch_h),
                    (100, h - 100 - patch_h, 100 + patch_w, h - 100),
                    (w - 100 - patch_w, h - 100 - patch_h, w - 100, h - 100),
                    (w // 2 - patch_w // 2, h // 2 - patch_h // 2, w // 2 + patch_w // 2, h // 2 + patch_h // 2)
                ]
                for box in crops:
                    patch = img.crop(box)
                    backgrounds.append(patch.copy())
        except Exception as e:
            print(f"Error reading {fname}: {e}")
            
    print(f"Extracted {len(backgrounds)} real ground backgrounds successfully.")
    return backgrounds

# --- 3. SYNTHETIC OVERLAY AND PERSPECTIVE WARPING ---
def warp_and_overlay_flag(bg_image, flag_image, class_id):
    bg_w, bg_h = bg_image.size
    fl_w, fl_h = flag_image.size
    
    # Scale flags to match validation frames
    scale = random.uniform(0.06, 0.22)
    new_w = int(fl_w * scale)
    new_h = int(fl_h * scale)
    
    # Apply mild pixelation / downsampling
    pixel_scale = random.uniform(0.70, 0.95)
    px_w = max(16, int(new_w * pixel_scale))
    px_h = max(12, int(new_h * pixel_scale))
    flag_px = flag_image.resize((px_w, px_h), Image.Resampling.LANCZOS)
    flag_scaled = flag_px.resize((new_w, new_h), Image.Resampling.NEAREST)
    
    border_px = max(1, int(new_w * 0.05))
    flag_bordered = Image.new('RGBA', (new_w + 2*border_px, new_h + 2*border_px), (255, 255, 255, 255))
    flag_bordered.paste(flag_scaled, (border_px, border_px))
    fw, fh = flag_bordered.size
    
    # Create an overlay for effects
    flag_effects = Image.new('RGBA', (fw, fh), (0, 0, 0, 0))
    draw_fl = ImageDraw.Draw(flag_effects)
    
    # Glare
    if random.random() < 0.6:
        gx = random.randint(-fw//2, fw + fw//2)
        gy = random.randint(-fh//2, fh + fh//2)
        gr = random.randint(min(fw, fh), max(fw, fh) * 2)
        for r_step in range(gr, 0, -5):
            alpha = int(random.uniform(2, 6) * (1.0 - r_step / gr) * 2.5)
            draw_fl.ellipse([gx - r_step, gy - r_step, gx + r_step, gy + r_step], fill=(255, 253, 245, alpha))
            
    # Dust
    if random.random() < 0.7:
        num_dust = random.randint(3, 8)
        for _ in range(num_dust):
            dx = random.randint(0, fw)
            dy = random.randint(0, fh)
            dr = random.randint(1, 3)
            alpha = random.randint(50, 150)
            draw_fl.ellipse([dx - dr, dy - dr, dx + dr, dy + dr], fill=(70, 60, 50, alpha))
            
    flag_alpha_channel = flag_bordered.split()[3]
    flag_composite = Image.composite(Image.alpha_composite(flag_bordered, flag_effects), flag_bordered, flag_alpha_channel)
    
    # Blur
    blur_radius = random.uniform(0.0, 0.3)
    if blur_radius > 0.05:
        flag_final_temp = flag_composite.filter(ImageFilter.GaussianBlur(blur_radius))
    else:
        flag_final_temp = flag_composite
    
    corners = np.array([
        [-fw/2, -fh/2, 1],
        [fw/2, -fh/2, 1],
        [fw/2, fh/2, 1],
        [-fw/2, fh/2, 1]
    ])
    
    theta = random.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])
    
    p_x = random.uniform(-0.25, 0.25) / fw
    p_y = random.uniform(-0.25, 0.25) / fh
    P = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [p_x, p_y, 1]
    ])
    
    T = P @ R
    warped_corners_h = (T @ corners.T).T
    warped_corners = warped_corners_h[:, :2] / warped_corners_h[:, 2:3]
    
    tx = random.uniform(100, bg_w - 100)
    ty = random.uniform(100, bg_h - 100)
    warped_corners[:, 0] += tx
    warped_corners[:, 1] += ty
    
    src_pts = np.array([[0, 0], [fw, 0], [fw, fh], [0, fh]], dtype=np.float32)
    dst_pts = warped_corners.astype(np.float32)
    H_mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
    
    flag_cv = cv2.cvtColor(np.array(flag_final_temp), cv2.COLOR_RGBA2BGRA)
    warped_flag_cv = cv2.warpPerspective(flag_cv, H_mat, (bg_w, bg_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    
    alpha = warped_flag_cv[:, :, 3]
    
    shadow_scale = scale / 0.20
    kernel_size = int(25 * shadow_scale)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(5, kernel_size)
    
    shadow_mask = cv2.GaussianBlur(alpha, (kernel_size, kernel_size), 0)
    sh_dx = int(random.uniform(4, 10) * shadow_scale)
    sh_dy = int(random.uniform(4, 10) * shadow_scale)
    M = np.float32([[1, 0, sh_dx], [0, 1, sh_dy]])
    shadow_mask_trans = cv2.warpAffine(shadow_mask, M, (bg_w, bg_h))
    
    bg_cv = cv2.cvtColor(np.array(bg_image), cv2.COLOR_RGB2BGR)
    shadow_intensity = random.uniform(0.20, 0.38)
    shadow_mask_norm = (shadow_mask_trans / 255.0) * shadow_intensity
    for channel in range(3):
        bg_cv[:, :, channel] = bg_cv[:, :, channel] * (1.0 - shadow_mask_norm)
        
    flag_rgb = warped_flag_cv[:, :, :3].copy()
    
    # 1. Color Tinting: blend flag colors to match ambient sandy light (10% to 35%)
    bg_mean = np.mean(bg_cv, axis=(0, 1))
    tint_factor = random.uniform(0.10, 0.35)
    mask = warped_flag_cv[:, :, 3] > 0
    for channel in range(3):
        flag_rgb[mask, channel] = flag_rgb[mask, channel] * (1.0 - tint_factor) + bg_mean[channel] * tint_factor
        
    # 2. Smooth alpha and apply faded GoPro Hero 13 opacity (78% to 92%)
    flag_alpha = cv2.GaussianBlur(warped_flag_cv[:, :, 3], (3, 3), 0) / 255.0
    opacity = random.uniform(0.78, 0.92)
    flag_alpha *= opacity
    
    for channel in range(3):
        bg_cv[:, :, channel] = flag_rgb[:, :, channel] * flag_alpha + bg_cv[:, :, channel] * (1.0 - flag_alpha)
        
    bg_final = Image.fromarray(cv2.cvtColor(bg_cv.astype(np.uint8), cv2.COLOR_BGR2RGB))
    
    x_coords = warped_corners[:, 0]
    y_coords = warped_corners[:, 1]
    x_min = max(0.0, np.min(x_coords))
    x_max = min(float(bg_w), np.max(x_coords))
    y_min = max(0.0, np.min(y_coords))
    y_max = min(float(bg_h), np.max(y_coords))
    
    cx = ((x_min + x_max) / 2.0) / bg_w
    cy = ((y_min + y_max) / 2.0) / bg_h
    bw = (x_max - x_min) / bg_w
    bh = (y_max - y_min) / bg_h
    
    yolo_label = f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
    return bg_final, yolo_label

def apply_drone_camera_effects(image):
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(random.uniform(0.85, 1.25))
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.uniform(0.85, 1.18))
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(random.uniform(0.82, 1.22))
    
    img_np = np.array(image)
    if random.random() < 0.3:
        noise = np.random.normal(0, random.uniform(1.0, 4.0), img_np.shape)
        img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(img_np)
        
    return image

# --- 4. DATASET GENERATION PIPELINE ---
def generate_dataset(num_train=3200, num_val=800):
    dataset_base_dir = r'C:\Users\mido\Documents\antigravity\focused-babbage\synthetic_dataset'
    print(f"Generating full dataset at: {dataset_base_dir}")
    
    for split in ['train', 'val']:
        os.makedirs(os.path.join(dataset_base_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(dataset_base_dir, 'labels', split), exist_ok=True)
        
    backgrounds = extract_backgrounds_from_frames()
    
    splits = [
        ('train', num_train),
        ('val', num_val)
    ]
    
    # Load all template images
    flag_templates = []
    for name in sorted_classes:
        path = template_paths[name]
        try:
            with Image.open(path) as img:
                img_rgba = img.convert('RGBA')
                base_w = 300
                base_h = int((base_w / img_rgba.size[0]) * img_rgba.size[1])
                img_resized = img_rgba.resize((base_w, base_h), Image.Resampling.LANCZOS)
                flag_templates.append(img_resized)
        except Exception as e:
            print(f"Error loading template {name} from {path}: {e}")
            raise e
            
    for split_name, count in splits:
        print(f"\nGenerating {count} images for split: {split_name}...")
        for i in range(count):
            bg = random.choice(backgrounds).copy()
            
            # 12% probability of generating a negative background sample
            is_negative = random.random() < 0.12
            
            if is_negative:
                img_final = apply_drone_camera_effects(bg)
                label_str = ""
            else:
                class_id = random.randint(0, len(flag_templates) - 1)
                flag_temp = flag_templates[class_id]
                img_blended, label_str = warp_and_overlay_flag(bg, flag_temp, class_id)
                img_final = apply_drone_camera_effects(img_blended)
                
            filename = f"flag_synth_{split_name}_{i:05d}"
            img_path = os.path.join(dataset_base_dir, 'images', split_name, f"{filename}.jpg")
            lbl_path = os.path.join(dataset_base_dir, 'labels', split_name, f"{filename}.txt")
            
            img_final.save(img_path, 'JPEG', quality=random.randint(70, 95))
            with open(lbl_path, 'w') as lf:
                if label_str:
                    lf.write(label_str + "\n")
                    
            if (i + 1) % 200 == 0 or i + 1 == count:
                print(f"  Processed {i+1} / {count} images...")
                
    # Build names mapping for dataset.yaml
    names_str = ""
    for idx, name in enumerate(sorted_classes):
        names_str += f"  {idx}: {name}\n"
        
    yaml_content = f"""path: {dataset_base_dir.replace('\\', '/')}
train: images/train
val: images/val

names:
{names_str}"""

    yaml_path = os.path.join(dataset_base_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as yf:
        yf.write(yaml_content)
        
    print(f"\nSynthetic dataset generation complete!")
    print(f"YOLO dataset config saved at: {yaml_path}")

if __name__ == '__main__':
    generate_dataset()
