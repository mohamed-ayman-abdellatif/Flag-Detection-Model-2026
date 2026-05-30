import os
import cv2
import random
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import multiprocessing

# Global Constants
APPEARANCES_PER_FLAG = 20
FLAGS_PER_IMAGE_LIMIT = 5

def generate_single_image(args):
    """
    Worker function to generate a single synthetic image with up to 5 unique non-overlapping flags.
    Utilizes localized warping (200x200px patches) to optimize CPU speed.
    """
    img_idx, selected_classes, frame_files, output_dir, sorted_classes, template_paths = args
    
    try:
        # 1. Load a random background frame
        bg_path = random.choice(frame_files)
        bg_image = Image.open(bg_path)
        
        # Ensure exact 4000x3000px resolution
        if bg_image.size != (4000, 3000):
            bg_image = bg_image.resize((4000, 3000), Image.Resampling.LANCZOS)
            
        bg_w, bg_h = bg_image.size
        bg_cv = cv2.cvtColor(np.array(bg_image), cv2.COLOR_RGB2BGR)
        
        # We will accumulate labels and mask overlays
        label_lines = []
        placed_positions = []
        
        for class_id in selected_classes:
            class_name = sorted_classes[class_id]
            flag_path = template_paths[class_name]
            
            # Load flag template
            with Image.open(flag_path) as img:
                flag_rgba = img.convert('RGBA')
                
            # Resize template to a base size for processing
            base_w = 300
            base_h = int((base_w / flag_rgba.size[0]) * flag_rgba.size[1])
            flag_rgba = flag_rgba.resize((base_w, base_h), Image.Resampling.LANCZOS)
            
            # Create flag board with white border
            border_px = max(1, int(base_w * 0.05))
            flag_bordered = Image.new('RGBA', (base_w + 2*border_px, base_h + 2*border_px), (255, 255, 255, 255))
            flag_bordered.paste(flag_rgba, (border_px, border_px))
            fw, fh = flag_bordered.size
            
            # Create subtle glare overlay
            flag_effects = Image.new('RGBA', (fw, fh), (0, 0, 0, 0))
            draw_fl = ImageDraw.Draw(flag_effects)
            if random.random() < 0.4:
                gx = random.randint(-fw//2, fw + fw//2)
                gy = random.randint(-fh//2, fh + fh//2)
                gr = random.randint(min(fw, fh), max(fw, fh) * 2)
                for r_step in range(gr, 0, -5):
                    alpha_gl = int(random.uniform(1, 4) * (1.0 - r_step / gr) * 2.5)
                    draw_fl.ellipse([gx - r_step, gy - r_step, gx + r_step, gy + r_step], fill=(255, 253, 245, alpha_gl))
            
            flag_alpha_channel = flag_bordered.split()[3]
            flag_composite = Image.composite(Image.alpha_composite(flag_bordered, flag_effects), flag_bordered, flag_alpha_channel)
            flag_composite = flag_composite.filter(ImageFilter.GaussianBlur(random.uniform(0.1, 0.35)))
            
            # 2. Geometry transformations (Warp & Perspective)
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
            
            x_min_c = np.min(warped_corners_centered[:, 0])
            x_max_c = np.max(warped_corners_centered[:, 0])
            y_min_c = np.min(warped_corners_centered[:, 1])
            y_max_c = np.max(warped_corners_centered[:, 1])
            w_box = x_max_c - x_min_c
            h_box = y_max_c - y_min_c
            
            # Resize so flag's bounding box is 40-50px in final image
            target_size = random.uniform(40.0, 50.0)
            scale = target_size / max(w_box, h_box)
            
            # 3. Placement with overlap checks (distance threshold >= 250px)
            tx, ty = None, None
            for _ in range(150):
                candidate_x = random.uniform(150, bg_w - 150)
                candidate_y = random.uniform(150, bg_h - 150)
                
                too_close = False
                for px, py in placed_positions:
                    dist = np.hypot(candidate_x - px, candidate_y - py)
                    if dist < 250.0:  # Safety margin to prevent any overlap
                        too_close = True
                        break
                if not too_close:
                    tx, ty = candidate_x, candidate_y
                    break
            
            if tx is None:
                tx = random.uniform(150, bg_w - 150)
                ty = random.uniform(150, bg_h - 150)
                
            placed_positions.append((tx, ty))
            
            # Project corners
            warped_corners = warped_corners_centered * scale
            warped_corners[:, 0] += tx
            warped_corners[:, 1] += ty
            
            # --- LOCAL WARP OPTIMIZATION ---
            # Define a local 200x200 patch bounds centered on (tx, ty)
            local_w, local_h = 200, 200
            x1 = int(tx - local_w / 2)
            y1 = int(ty - local_h / 2)
            # Clip bounds
            x1 = max(0, min(x1, bg_w - local_w))
            y1 = max(0, min(y1, bg_h - local_h))
            x2 = x1 + local_w
            y2 = y1 + local_h
            
            # Shift destination points to local patch coordinates
            dst_pts_local = warped_corners.astype(np.float32) - np.array([x1, y1], dtype=np.float32)
            src_pts = np.array([[0, 0], [fw, 0], [fw, fh], [0, fh]], dtype=np.float32)
            H_mat_local = cv2.getPerspectiveTransform(src_pts, dst_pts_local)
            
            # Warp flag into local 200x200 canvas
            flag_cv = cv2.cvtColor(np.array(flag_composite), cv2.COLOR_RGBA2BGRA)
            warped_flag_local = cv2.warpPerspective(flag_cv, H_mat_local, (local_w, local_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
            
            alpha_local = warped_flag_local[:, :, 3]
            
            # Local shadow rendering
            shadow_intensity = random.uniform(0.18, 0.35)
            shadow_mask_local = cv2.GaussianBlur(alpha_local, (random.choice([3, 5, 7]), random.choice([3, 5, 7])), 0)
            sh_dx = int(random.uniform(2, 5))
            sh_dy = int(random.uniform(2, 5))
            M_sh = np.float32([[1, 0, sh_dx], [0, 1, sh_dy]])
            shadow_mask_trans_local = cv2.warpAffine(shadow_mask_local, M_sh, (local_w, local_h))
            shadow_mask_norm_local = (shadow_mask_trans_local / 255.0) * shadow_intensity
            
            # Apply shadow to local patch
            local_bg = bg_cv[y1:y2, x1:x2].copy()
            for channel in range(3):
                local_bg[:, :, channel] = local_bg[:, :, channel] * (1.0 - shadow_mask_norm_local)
                
            # Local lighting desaturation & tinting
            flag_rgb_local = warped_flag_local[:, :, :3].copy()
            bg_mean = np.mean(local_bg, axis=(0, 1))
            
            tint_factor = random.uniform(0.08, 0.22)
            mask_flag_local = alpha_local > 0
            for channel in range(3):
                flag_rgb_local[mask_flag_local, channel] = flag_rgb_local[mask_flag_local, channel] * (1.0 - tint_factor) + bg_mean[channel] * tint_factor
                
            # Local alpha blending
            flag_alpha_local = cv2.GaussianBlur(alpha_local, (3, 3), 0) / 255.0
            opacity = random.uniform(0.78, 0.92)
            flag_alpha_local *= opacity
            
            for channel in range(3):
                local_bg[:, :, channel] = flag_rgb_local[:, :, channel] * flag_alpha_local + local_bg[:, :, channel] * (1.0 - flag_alpha_local)
                
            # Write back the blended patch
            bg_cv[y1:y2, x1:x2] = local_bg
            # -------------------------------
            
            # Compute YOLO bbox values
            final_x_min = max(0.0, np.min(warped_corners[:, 0]))
            final_x_max = min(float(bg_w), np.max(warped_corners[:, 0]))
            final_y_min = max(0.0, np.min(warped_corners[:, 1]))
            final_y_max = min(float(bg_h), np.max(warped_corners[:, 1]))
            
            final_w = final_x_max - final_x_min
            final_h = final_y_max - final_y_min
            
            cx = ((final_x_min + final_x_max) / 2.0) / bg_w
            cy = ((final_y_min + final_y_max) / 2.0) / bg_h
            norm_w = final_w / bg_w
            norm_h = final_h / bg_h
            
            label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}")
            
        # Convert back to PIL Image and apply general camera lens effects
        img_final = Image.fromarray(cv2.cvtColor(bg_cv.astype(np.uint8), cv2.COLOR_BGR2RGB))
        
        enhancer = ImageEnhance.Color(img_final)
        img_final = enhancer.enhance(random.uniform(0.92, 1.15))
        enhancer = ImageEnhance.Brightness(img_final)
        img_final = enhancer.enhance(random.uniform(0.95, 1.08))
        
        # Subtle sensor Gaussian noise/blur
        if random.random() < 0.4:
            img_final = img_final.filter(ImageFilter.GaussianBlur(random.uniform(0.1, 0.35)))
            
        # Save output image
        out_name = f"multi_flag_{img_idx:05d}.jpg"
        out_path = os.path.join(output_dir, out_name)
        img_final.save(out_path, 'JPEG', quality=random.randint(85, 95))
        
        # Save output labels text file
        lbl_name = f"multi_flag_{img_idx:05d}.txt"
        lbl_path = os.path.join(output_dir, lbl_name)
        with open(lbl_path, 'w') as lf:
            for line in label_lines:
                lf.write(line + "\n")
                
        return True, img_idx, None
        
    except Exception as e:
        return False, img_idx, str(e)


def main():
    # Setup paths
    ref_country_dir = r"C:\Users\mido\Documents\antigravity\focused-babbage\reference\country"
    ref_inst_dir = r"C:\Users\mido\Documents\antigravity\focused-babbage\reference\institution"
    frames_dir = r"c:\Users\mido\Downloads\frames-20260515T124041Z-3-001\frames"
    output_dir = r"C:\Users\mido\Documents\antigravity\focused-babbage\validate_multi_flag"
    
    print("=== Multi-Flag Synthetic Dataset Generator (Resumable, Multiprocessing, Option C, Local Workspace) ===")
    
    # Verify directories
    if not os.path.exists(ref_country_dir):
        print(f"Error: country flags reference directory not found at {ref_country_dir}")
        return
    if not os.path.exists(ref_inst_dir):
        print(f"Error: institution flags reference directory not found at {ref_inst_dir}")
        return
    if not os.path.exists(frames_dir):
        print(f"Error: background frames not found at {frames_dir}")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. LOAD ALL TEMPLATES DYNAMICALLY
    template_paths = {}
    
    # Countries (PNG)
    for path in glob.glob(os.path.join(ref_country_dir, "*")):
        if path.lower().endswith(".png"):
            name = os.path.basename(path).replace(".png", "").replace(".PNG", "")
            template_paths[name] = path
            
    # Institutions (PNG, JPG, JPEG, GIF)
    for path in glob.glob(os.path.join(ref_inst_dir, "*")):
        if path.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
            name = os.path.basename(path).split(".")[0]
            template_paths[name] = path
            
    # Sort class names alphabetically
    sorted_classes = sorted(template_paths.keys())
    class_to_id = {name: idx for idx, name in enumerate(sorted_classes)}
    
    print(f"Total dynamic flag classes loaded: {len(sorted_classes)}")
    
    # 2. CHECK RESUME STATUS
    # We will find how many occurrences of each class have already been generated
    generated_counts = {idx: 0 for idx in range(len(sorted_classes))}
    existing_txt_files = glob.glob(os.path.join(output_dir, "multi_flag_*.txt"))
    
    print(f"Scanning output folder for existing files to resume...")
    max_img_idx = 0
    for txt_path in existing_txt_files:
        base = os.path.basename(txt_path)
        try:
            idx_part = int(base.replace("multi_flag_", "").replace(".txt", ""))
            max_img_idx = max(max_img_idx, idx_part)
        except ValueError:
            continue
            
        with open(txt_path, 'r') as tf:
            for line in tf:
                line = line.strip()
                if line:
                    parts = line.split()
                    class_id = int(parts[0])
                    if class_id in generated_counts:
                        generated_counts[class_id] += 1
                        
    print(f"Found {len(existing_txt_files)} existing multi-flag image files. Highest index: {max_img_idx}")
    
    # 3. BUILD THE REMAINING PLACEMENT SCHEDULE
    flag_pool = []
    for class_id in range(len(sorted_classes)):
        needed = max(0, APPEARANCES_PER_FLAG - generated_counts[class_id])
        flag_pool.extend([class_id] * needed)
        
    random.seed(42)  # Seed for reproducibility of schedules
    random.shuffle(flag_pool)
    
    if len(flag_pool) == 0:
        print("Dataset is already complete! All 319 flags have 20+ appearances.")
        return
        
    images_to_generate = []
    while len(flag_pool) > 0:
        n_flags = min(FLAGS_PER_IMAGE_LIMIT, len(flag_pool))
        
        selected_classes = []
        indices_to_remove = []
        
        for idx, class_id in enumerate(flag_pool):
            if class_id not in selected_classes:
                selected_classes.append(class_id)
                indices_to_remove.append(idx)
                if len(selected_classes) == n_flags:
                    break
                    
        for idx in sorted(indices_to_remove, reverse=True):
            flag_pool.pop(idx)
            
        images_to_generate.append(selected_classes)
        
    n_images_to_gen = len(images_to_generate)
    total_images_target = max_img_idx + n_images_to_gen
    print(f"Remaining images to generate: {n_images_to_gen} (will bring total dataset to {total_images_target} images)")
    
    # 4. LOAD AND FILTER BACKGROUND FRAMES (First 400, Sand Only)
    all_raw_frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))[:400]
    print(f"Scanning and filtering first 400 frames for pavement...")
    
    frame_files = []
    for path in all_raw_frames:
        img = cv2.imread(path)
        if img is not None:
            # Downsample for ultra-fast color analysis
            img_small = cv2.resize(img, (100, 75))
            hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
            
            # Pavement is low-saturation gray (S < 45) and not pitch black (V > 50)
            gray_mask = (hsv[:, :, 1] < 45) & (hsv[:, :, 2] > 50)
            gray_ratio = np.mean(gray_mask)
            
            if gray_ratio < 0.25:  # Keep only images with less than 25% grey area
                frame_files.append(path)
                
    print(f"Pavement filtering complete. Kept {len(frame_files)} / {len(all_raw_frames)} pure sand background frames.")
    if not frame_files:
        print("Error: No valid background frames found after pavement filtering!")
        return
        
    # 5. INITIALIZE MULTIPROCESSING
    cpu_count = multiprocessing.cpu_count()
    num_workers = max(1, cpu_count - 1)
    print(f"Running multiprocessing with {num_workers} parallel workers...")
    
    tasks = []
    for i, selected_classes in enumerate(images_to_generate):
        tasks.append((max_img_idx + i + 1, selected_classes, frame_files, output_dir, sorted_classes, template_paths))
        
    pool = multiprocessing.Pool(processes=num_workers)
    
    completed = 0
    errors = []
    
    for success, img_idx, error_msg in pool.imap_unordered(generate_single_image, tasks):
        completed += 1
        if not success:
            errors.append((img_idx, error_msg))
            
        if completed % 100 == 0 or completed == n_images_to_gen:
            print(f"  Completed {completed} / {n_images_to_gen} remaining images...")
            
    pool.close()
    pool.join()
    
    print(f"\nMultiprocessing pool finished. Generated: {completed} new images.")
    if errors:
        print(f"Encountered {len(errors)} errors during generation:")
        for img_idx, err in errors[:10]:
            print(f"  Image {img_idx}: {err}")
            
    # Save YOLO config file
    names_str = ""
    for idx, name in enumerate(sorted_classes):
        names_str += f"  {idx}: {name}\n"
        
    yaml_content = f"""path: {output_dir.replace('\\', '/')}
train: .
val: .

names:
{names_str}"""

    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, 'w') as yf:
        yf.write(yaml_content)
    print(f"Saved dataset config to: {yaml_path}")
    
    # 6. RUN SELF-VERIFICATION
    print("\n=== Running Self-Verification ===")
    generated_jpgs = glob.glob(os.path.join(output_dir, "*.jpg"))
    print(f"Verified count of JPEGs: {len(generated_jpgs)}")
    
    bad_res_count = 0
    bad_size_count = 0
    overlap_count = 0
    total_annotations_checked = 0
    
    for path in generated_jpgs:
        with Image.open(path) as check_img:
            if check_img.size != (4000, 3000):
                bad_res_count += 1
                
        txt_path = path.replace(".jpg", ".txt")
        bboxes = []
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as tf:
                lines = tf.readlines()
                for line in lines:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        w_norm = float(parts[3])
                        h_norm = float(parts[4])
                        w_px = w_norm * 4000
                        h_px = h_norm * 3000
                        max_dim = max(w_px, h_px)
                        total_annotations_checked += 1
                        if not (38.0 <= max_dim <= 52.0):
                            bad_size_count += 1
                            
                        cx_norm = float(parts[1])
                        cy_norm = float(parts[2])
                        cx_px = cx_norm * 4000
                        cy_px = cy_norm * 3000
                        bboxes.append((cx_px, cy_px, w_px, h_px))
                        
            for a in range(len(bboxes)):
                for b in range(a + 1, len(bboxes)):
                    cx_a, cy_a, w_a, h_a = bboxes[a]
                    cx_b, cy_b, w_b, h_b = bboxes[b]
                    dist = np.hypot(cx_a - cx_b, cy_a - cy_b)
                    if dist < 120.0:
                        overlap_count += 1
                        
    print(f"Total flag annotations checked: {total_annotations_checked}")
    print(f"Images with incorrect resolution (!= 4000x3000): {bad_res_count}")
    print(f"Individual flag annotations out of size bounds (38px - 52px): {bad_size_count}")
    print(f"Flag overlap incidents: {overlap_count}")
    
    if len(generated_jpgs) == total_images_target and bad_res_count == 0 and bad_size_count == 0 and overlap_count == 0:
        print("SUCCESS: Multi-flag dataset matches all criteria perfectly!")
    else:
        print("WARNING: Self-verification flagged some discrepancies. Please check details.")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
