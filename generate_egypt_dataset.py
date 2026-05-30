import os
import cv2
import random
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

def generate_egypt_dataset():
    # 1. SETUP PATHS
    egypt_flag_path = r"C:\Users\mido\Documents\antigravity\focused-babbage\reference\country\eg.png"
    frames_dir = r"c:\Users\mido\Downloads\frames-20260515T124041Z-3-001\frames"
    output_dir = r"C:\Users\mido\Documents\antigravity\focused-babbage\validate_egypt"
    
    print("=== Egyptian Flag Synthetic Dataset Generator ===")
    print(f"Flag template: {egypt_flag_path}")
    print(f"Background frames directory: {frames_dir}")
    print(f"Output directory: {output_dir}")
    
    # 2. VALIDATE INPUTS
    if not os.path.exists(egypt_flag_path):
        print(f"Error: Egypt flag template not found at {egypt_flag_path}")
        return
        
    if not os.path.exists(frames_dir):
        print(f"Error: Background frames directory not found at {frames_dir}")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Egypt flag image
    with Image.open(egypt_flag_path) as img:
        flag_rgba = img.convert('RGBA')
        
    # Scale flag template to a reasonable base size for processing
    base_w = 300
    base_h = int((base_w / flag_rgba.size[0]) * flag_rgba.size[1])
    flag_rgba = flag_rgba.resize((base_w, base_h), Image.Resampling.LANCZOS)
    
    # Get all background frames
    frame_files = glob.glob(os.path.join(frames_dir, "*.jpg"))
    if not frame_files:
        print("Error: No background frames (*.jpg) found!")
        return
    print(f"Found {len(frame_files)} background frames.")
    
    # Generate 200 images
    n_images = 200
    annotations = []
    
    for i in range(1, n_images + 1):
        bg_path = random.choice(frame_files)
        bg_image = Image.open(bg_path)
        
        # Ensure exact 4000x3000px resolution
        if bg_image.size != (4000, 3000):
            bg_image = bg_image.resize((4000, 3000), Image.Resampling.LANCZOS)
            
        bg_w, bg_h = bg_image.size
        
        # Create flag board with white border (similar to real validation targets printed on white cards)
        # Bounding box border represents white paper/card margin
        border_px = max(1, int(base_w * 0.05))
        flag_bordered = Image.new('RGBA', (base_w + 2*border_px, base_h + 2*border_px), (255, 255, 255, 255))
        flag_bordered.paste(flag_rgba, (border_px, border_px))
        fw, fh = flag_bordered.size
        
        # Create overlay effects for dust/glare on the flag board
        flag_effects = Image.new('RGBA', (fw, fh), (0, 0, 0, 0))
        draw_fl = ImageDraw.Draw(flag_effects)
        
        # Optional subtle glare
        if random.random() < 0.5:
            gx = random.randint(-fw//2, fw + fw//2)
            gy = random.randint(-fh//2, fh + fh//2)
            gr = random.randint(min(fw, fh), max(fw, fh) * 2)
            for r_step in range(gr, 0, -5):
                alpha = int(random.uniform(1, 4) * (1.0 - r_step / gr) * 2.5)
                draw_fl.ellipse([gx - r_step, gy - r_step, gx + r_step, gy + r_step], fill=(255, 253, 245, alpha))
                
        # Combine flag and effects
        flag_alpha_channel = flag_bordered.split()[3]
        flag_composite = Image.composite(Image.alpha_composite(flag_bordered, flag_effects), flag_bordered, flag_alpha_channel)
        
        # Mild blur on the high-res flag template
        flag_composite = flag_composite.filter(ImageFilter.GaussianBlur(random.uniform(0.1, 0.4)))
        
        # Define base corner coordinates relative to center
        corners = np.array([
            [-fw/2, -fh/2, 1],
            [fw/2, -fh/2, 1],
            [fw/2, fh/2, 1],
            [-fw/2, fh/2, 1]
        ])
        
        # Random rotation (0 to 360 degrees)
        theta = random.uniform(0, 2 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
        
        # Random mild perspective warp
        p_x = random.uniform(-0.15, 0.15) / fw
        p_y = random.uniform(-0.15, 0.15) / fh
        P = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [p_x, p_y, 1]
        ])
        
        T = P @ R
        warped_corners_h = (T @ corners.T).T
        warped_corners_centered = warped_corners_h[:, :2] / warped_corners_h[:, 2:3]
        
        # Calculate bounding box dimensions of unscaled warped shape
        x_min_c = np.min(warped_corners_centered[:, 0])
        x_max_c = np.max(warped_corners_centered[:, 0])
        y_min_c = np.min(warped_corners_centered[:, 1])
        y_max_c = np.max(warped_corners_centered[:, 1])
        w_box = x_max_c - x_min_c
        h_box = y_max_c - y_min_c
        
        # Target flag size in dataset is 40-50px (bounding box maximum dimension)
        target_size = random.uniform(40.0, 50.0)
        scale = target_size / max(w_box, h_box)
        
        # Random paste location (avoiding margins)
        tx = random.uniform(150, bg_w - 150)
        ty = random.uniform(150, bg_h - 150)
        
        # Apply scaling and translation
        warped_corners = warped_corners_centered * scale
        warped_corners[:, 0] += tx
        warped_corners[:, 1] += ty
        
        # Compute perspective matrix
        src_pts = np.array([[0, 0], [fw, 0], [fw, fh], [0, fh]], dtype=np.float32)
        dst_pts = warped_corners.astype(np.float32)
        H_mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Warp flag image
        flag_cv = cv2.cvtColor(np.array(flag_composite), cv2.COLOR_RGBA2BGRA)
        warped_flag_cv = cv2.warpPerspective(flag_cv, H_mat, (bg_w, bg_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
        
        alpha = warped_flag_cv[:, :, 3]
        
        # Generate shadow (offset down and right, blurred)
        shadow_intensity = random.uniform(0.18, 0.35)
        shadow_mask = cv2.GaussianBlur(alpha, (random.choice([3, 5, 7]), random.choice([3, 5, 7])), 0)
        sh_dx = int(random.uniform(2, 5))
        sh_dy = int(random.uniform(2, 5))
        M = np.float32([[1, 0, sh_dx], [0, 1, sh_dy]])
        shadow_mask_trans = cv2.warpAffine(shadow_mask, M, (bg_w, bg_h))
        shadow_mask_norm = (shadow_mask_trans / 255.0) * shadow_intensity
        
        # Apply shadow to background
        bg_cv = cv2.cvtColor(np.array(bg_image), cv2.COLOR_RGB2BGR)
        for channel in range(3):
            bg_cv[:, :, channel] = bg_cv[:, :, channel] * (1.0 - shadow_mask_norm)
            
        # Color tint flag colors based on local terrain mean color
        flag_rgb = warped_flag_cv[:, :, :3].copy()
        local_y_min, local_y_max = max(0, int(ty)-80), min(bg_h, int(ty)+80)
        local_x_min, local_x_max = max(0, int(tx)-80), min(bg_w, int(tx)+80)
        local_bg = bg_cv[local_y_min:local_y_max, local_x_min:local_x_max]
        
        if local_bg.size > 0:
            bg_mean = np.mean(local_bg, axis=(0, 1))
        else:
            bg_mean = np.array([120.0, 150.0, 180.0]) # fallback
            
        tint_factor = random.uniform(0.08, 0.22)
        mask = alpha > 0
        for channel in range(3):
            flag_rgb[mask, channel] = flag_rgb[mask, channel] * (1.0 - tint_factor) + bg_mean[channel] * tint_factor
            
        # Soften alpha edges and apply GoPro lens fading/opacity
        flag_alpha = cv2.GaussianBlur(alpha, (3, 3), 0) / 255.0
        opacity = random.uniform(0.80, 0.93)
        flag_alpha *= opacity
        
        # Blend flag onto background
        for channel in range(3):
            bg_cv[:, :, channel] = flag_rgb[:, :, channel] * flag_alpha + bg_cv[:, :, channel] * (1.0 - flag_alpha)
            
        # Convert back to PIL Image
        img_blended = Image.fromarray(cv2.cvtColor(bg_cv.astype(np.uint8), cv2.COLOR_BGR2RGB))
        
        # Apply camera effects: mild color saturation, brightness, contrast perturbations
        enhancer = ImageEnhance.Color(img_blended)
        img_blended = enhancer.enhance(random.uniform(0.92, 1.15))
        enhancer = ImageEnhance.Brightness(img_blended)
        img_blended = enhancer.enhance(random.uniform(0.95, 1.08))
        
        # Apply very subtle Gaussian noise/blur to the full frame
        if random.random() < 0.4:
            img_blended = img_blended.filter(ImageFilter.GaussianBlur(random.uniform(0.1, 0.35)))
            
        # Calculate final bounding box coordinates in background coordinate space
        final_x_min = max(0.0, np.min(warped_corners[:, 0]))
        final_x_max = min(float(bg_w), np.max(warped_corners[:, 0]))
        final_y_min = max(0.0, np.min(warped_corners[:, 1]))
        final_y_max = min(float(bg_h), np.max(warped_corners[:, 1]))
        
        final_w = final_x_max - final_x_min
        final_h = final_y_max - final_y_min
        
        # Save output image
        out_name = f"eg_{i:03d}.jpg"
        out_path = os.path.join(output_dir, out_name)
        img_blended.save(out_path, 'JPEG', quality=random.randint(85, 95))
        
        # YOLO coordinates: normalized cx, cy, w, h
        cx = ((final_x_min + final_x_max) / 2.0) / bg_w
        cy = ((final_y_min + final_y_max) / 2.0) / bg_h
        norm_w = final_w / bg_w
        norm_h = final_h / bg_h
        
        # Save label text file next to image
        lbl_name = f"eg_{i:03d}.txt"
        lbl_path = os.path.join(output_dir, lbl_name)
        with open(lbl_path, 'w') as lf:
            lf.write(f"0 {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}\n")
            
        annotations.append({
            'filename': out_name,
            'x_min': int(final_x_min),
            'y_min': int(final_y_min),
            'width': int(final_w),
            'height': int(final_h),
            'cx_yolo': cx,
            'cy_yolo': cy,
            'w_yolo': norm_w,
            'h_yolo': norm_h
        })
        
        if i % 25 == 0 or i == n_images:
            print(f"  Generated {i} / {n_images} images...")
            
    # Save a CSV label summary in the output folder for convenience
    csv_path = os.path.join(output_dir, "labels_summary.csv")
    with open(csv_path, 'w') as cf:
        cf.write("filename,x_min,y_min,width,height,cx_yolo,cy_yolo,w_yolo,h_yolo\n")
        for ann in annotations:
            cf.write(f"{ann['filename']},{ann['x_min']},{ann['y_min']},{ann['width']},{ann['height']},{ann['cx_yolo']:.6f},{ann['cy_yolo']:.6f},{ann['w_yolo']:.6f},{ann['h_yolo']:.6f}\n")
            
    # Save a dataset.yaml config inside the directory
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    yaml_content = f"""path: {output_dir.replace('\\', '/')}
train: .
val: .

names:
  0: eg
"""
    with open(yaml_path, 'w') as yf:
        yf.write(yaml_content)
        
    print("\nDataset generation complete!")
    print(f"Images and YOLO labels saved in: {output_dir}")
    print(f"Summary CSV saved to: {csv_path}")
    print(f"YOLO dataset config saved to: {yaml_path}")
    
    # 3. SELF-VERIFICATION
    print("\n=== Running Self-Verification ===")
    generated_jpgs = glob.glob(os.path.join(output_dir, "*.jpg"))
    print(f"Verified count of JPEGs: {len(generated_jpgs)}")
    
    bad_res_count = 0
    bad_size_count = 0
    
    for path in generated_jpgs:
        # Check resolution
        with Image.open(path) as check_img:
            if check_img.size != (4000, 3000):
                bad_res_count += 1
                
        # Check flag size in annotation
        txt_path = path.replace(".jpg", ".txt")
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as tf:
                line = tf.readline().strip()
                if line:
                    parts = line.split()
                    w_norm = float(parts[3])
                    h_norm = float(parts[4])
                    # convert back to pixels
                    w_px = w_norm * 4000
                    h_px = h_norm * 3000
                    max_dim = max(w_px, h_px)
                    if not (38.0 <= max_dim <= 52.0):
                        bad_size_count += 1
                        
    print(f"Images with incorrect resolution (!= 4000x3000): {bad_res_count}")
    print(f"Images with flag size out of bounds (38px - 52px): {bad_size_count}")
    
    if len(generated_jpgs) == 200 and bad_res_count == 0 and bad_size_count == 0:
        print("SUCCESS: Dataset matches all criteria perfectly!")
    else:
        print("WARNING: Some checks failed. Please check parameters.")

if __name__ == '__main__':
    generate_egypt_dataset()
