"""
generate_1class_dataset.py
Generates a synthetic dataset with a SINGLE class 'flag' (class 0).
All reference templates are used but all labeled as class 0.
This trains a fast flag *detector* (any flag vs background).
Classification into specific flag type is done post-detection via HSV matching.
"""

import os
import cv2
import glob
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

# ─── Reference templates ────────────────────────────────────────────────────
ref_country_dir = r"C:\Users\mido\Documents\antigravity\focused-babbage\reference\country"
ref_inst_dir    = r"C:\Users\mido\Documents\antigravity\focused-babbage\reference\institution"

template_paths = {}
for path in glob.glob(os.path.join(ref_country_dir, "*")):
    if path.lower().endswith(".png"):
        name = os.path.splitext(os.path.basename(path))[0]
        template_paths[name] = path
for path in glob.glob(os.path.join(ref_inst_dir, "*")):
    if path.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
        name = os.path.splitext(os.path.basename(path))[0]
        template_paths[name] = path

print(f"Templates loaded: {len(template_paths)}")

# ─── Backgrounds ─────────────────────────────────────────────────────────────
def extract_backgrounds():
    frames_dir = r"c:\Users\mido\Downloads\frames-20260515T124041Z-3-001\frames"
    sz = 640
    bgs = []
    
    # 1. Load original frames from downloads folder if they exist
    if os.path.exists(frames_dir):
        files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
        # Sample from 150 frames evenly distributed across the entire flight video
        idx_list = np.linspace(0, len(files)-1, min(150, len(files)), dtype=int)
        for idx in idx_list:
            p = os.path.join(frames_dir, files[idx])
            try:
                with Image.open(p) as img:
                    w, h = img.size
                    for (x0, y0) in [
                        (100, 100),
                        (w-100-sz, 100),
                        (100, h-100-sz),
                        (w-100-sz, h-100-sz),
                        (w//2-sz//2, h//2-sz//2),
                    ]:
                        x0 = max(0, min(x0, w-sz))
                        y0 = max(0, min(y0, h-sz))
                        bgs.append(img.crop((x0, y0, x0+sz, y0+sz)).copy())
            except Exception:
                pass
        print(f"Extracted {len(bgs)} background patches of size {sz}x{sz} from original frames.")
    else:
        # Fallback if frames directory doesn't exist
        for _ in range(50):
            bg = Image.new('RGB', (sz, sz), (218, 187, 156))
            arr = np.array(bg)
            noise = np.random.randint(-15, 15, arr.shape, dtype=np.int16)
            arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            bgs.append(Image.fromarray(arr))
            
    # 2. Load false positive negatives from 'new_negatives' folder if it exists
    new_neg_dir = r"C:\Users\mido\Documents\antigravity\focused-babbage\new_negatives"
    if os.path.exists(new_neg_dir):
        new_neg_files = glob.glob(os.path.join(new_neg_dir, "*.jpg"))
        if new_neg_files:
            print(f"Loading {len(new_neg_files)} custom desaturated false-positive negatives...")
            # We add them multiple times to increase their sampling weight (5x oversampling)
            for _ in range(5):
                for p in new_neg_files:
                    try:
                        with Image.open(p) as img:
                            bgs.append(img.copy())
                    except Exception:
                        pass
            print(f"Total backgrounds database size: {len(bgs)}")
            
    return bgs

# ─── Augmentation helpers ────────────────────────────────────────────────────
def camera_effects(image):
    image = ImageEnhance.Color(image).enhance(random.uniform(0.85, 1.25))
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.20))
    arr = np.array(image)
    if random.random() < 0.20:
        noise = np.random.normal(0, random.uniform(0.5, 2.0), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(arr)
    return image

def overlay_flag(bg_image, flag_image):
    """Overlay flag onto background, return (result_image, yolo_label_str)."""
    bg_w, bg_h = bg_image.size
    fl_w, fl_h = flag_image.size

    # Scale to 4–18% of image width (to cover extremely small/medium flags)
    scale = random.uniform(0.04, 0.18)
    new_w = int(fl_w * scale)
    new_h = int(fl_h * scale)

    # ASPECT RATIO PRESERVATION: Lock aspect ratio when applying minimum dimensions
    if new_w < 16:
        new_w = 16
        new_h = int(fl_h * (16.0 / fl_w))
    if new_h < 6:
        new_h = 6
        new_w = int(fl_w * (6.0 / fl_h))

    # Slight pixelation
    px_scale = random.uniform(0.75, 0.95)
    px_w, px_h = max(6, int(new_w * px_scale)), max(5, int(new_h * px_scale))
    flag_px = flag_image.resize((px_w, px_h), Image.Resampling.LANCZOS)
    flag_sc = flag_px.resize((new_w, new_h), Image.Resampling.NEAREST)

    # White border
    bp = max(1, int(new_w * 0.05))
    bordered = Image.new('RGBA', (new_w + 2*bp, new_h + 2*bp), (255, 255, 255, 255))
    bordered.paste(flag_sc, (bp, bp))
    fw, fh = bordered.size

    # Effects layer
    eff = Image.new('RGBA', (fw, fh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(eff)
    if random.random() < 0.5:
        gx = random.randint(-fw//2, fw + fw//2)
        gy = random.randint(-fh//2, fh + fh//2)
        gr = random.randint(min(fw,fh), max(fw,fh)*2)
        for r in range(gr, 0, -5):
            a = int(random.uniform(2, 5) * (1 - r/gr) * 2.5)
            draw.ellipse([gx-r, gy-r, gx+r, gy+r], fill=(255, 253, 245, a))
    if random.random() < 0.6:
        for _ in range(random.randint(2, 6)):
            dx, dy = random.randint(0, fw), random.randint(0, fh)
            dr = random.randint(1, 2)
            draw.ellipse([dx-dr, dy-dr, dx+dr, dy+dr], fill=(70, 60, 50, random.randint(30,120)))

    flag_alpha = bordered.split()[3]
    comp = Image.composite(Image.alpha_composite(bordered, eff), bordered, flag_alpha)
    
    # Reduced blur range to keep flags sharper
    blur_r = random.uniform(0.0, 0.20)
    if blur_r > 0.05:
        comp = comp.filter(ImageFilter.GaussianBlur(blur_r))

    # Perspective warp via OpenCV
    corners = np.array([[-fw/2,-fh/2,1],[fw/2,-fh/2,1],[fw/2,fh/2,1],[-fw/2,fh/2,1]])
    theta = random.uniform(0, 2*np.pi)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c,-s,0],[s,c,0],[0,0,1]])
    px = random.uniform(-0.20, 0.20)/fw
    py = random.uniform(-0.20, 0.20)/fh
    P = np.array([[1,0,0],[0,1,0],[px,py,1]])
    T = P @ R
    wh = (T @ corners.T).T
    wc = wh[:,:2] / wh[:,2:3]
    tx = random.uniform(50, bg_w-50)
    ty = random.uniform(50, bg_h-50)
    wc[:,0] += tx; wc[:,1] += ty

    src = np.float32([[0,0],[fw,0],[fw,fh],[0,fh]])
    dst = wc.astype(np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    fc = cv2.cvtColor(np.array(comp), cv2.COLOR_RGBA2BGRA)
    wf = cv2.warpPerspective(fc, H, (bg_w, bg_h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

    alpha = wf[:,:,3]
    # Shadow
    ks = max(5, int(25*(scale/0.20)))
    if ks % 2 == 0: ks += 1
    sm = cv2.GaussianBlur(alpha, (ks,ks), 0)
    sdx = int(random.uniform(3,8)*(scale/0.20))
    sdy = int(random.uniform(3,8)*(scale/0.20))
    M_s = np.float32([[1,0,sdx],[0,1,sdy]])
    smt = cv2.warpAffine(sm, M_s, (bg_w, bg_h))
    bg_cv = cv2.cvtColor(np.array(bg_image), cv2.COLOR_RGB2BGR)
    si = random.uniform(0.15, 0.30)
    smn = (smt/255.0)*si
    for ch in range(3):
        bg_cv[:,:,ch] = bg_cv[:,:,ch] * (1.0 - smn)

    # Color tinting (reduced range to prevent flag colors from completely fading)
    flag_rgb = wf[:,:,:3].copy()
    bg_mean = np.mean(bg_cv, axis=(0,1))
    tf = random.uniform(0.05, 0.20)
    mask = wf[:,:,3] > 0
    for ch in range(3):
        flag_rgb[mask,ch] = flag_rgb[mask,ch]*(1-tf) + bg_mean[ch]*tf

    # Alpha blend with high opacity
    fa = cv2.GaussianBlur(wf[:,:,3], (3,3), 0)/255.0
    opacity = random.uniform(0.85, 0.98)
    fa *= opacity
    for ch in range(3):
        bg_cv[:,:,ch] = flag_rgb[:,:,ch]*fa + bg_cv[:,:,ch]*(1-fa)

    result = Image.fromarray(cv2.cvtColor(bg_cv.astype(np.uint8), cv2.COLOR_BGR2RGB))

    # YOLO label (class 0 = flag)
    xs = wc[:,0]; ys = wc[:,1]
    xmin = max(0.0, float(xs.min()))
    xmax = min(float(bg_w), float(xs.max()))
    ymin = max(0.0, float(ys.min()))
    ymax = min(float(bg_h), float(ys.max()))
    cx = ((xmin+xmax)/2)/bg_w
    cy = ((ymin+ymax)/2)/bg_h
    bw = (xmax-xmin)/bg_w
    bh = (ymax-ymin)/bg_h
    label = f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
    return result, label


# ─── Dataset generation ───────────────────────────────────────────────────────
def generate_dataset(num_train=3000, num_val=800):
    base = r'C:\Users\mido\Documents\antigravity\focused-babbage\synthetic_dataset'
    for split in ['train', 'val']:
        os.makedirs(os.path.join(base, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(base, 'labels', split), exist_ok=True)

    # Delete old caches
    for cache in glob.glob(os.path.join(base, 'labels', '*.cache')):
        try: os.remove(cache)
        except: pass

    backgrounds = extract_backgrounds()

    # Load templates
    templates = []
    qatar_templates = []
    failing_templates = []
    
    # Read failing classes if file exists
    failing_names = set()
    failing_txt_path = r"C:\Users\mido\Documents\antigravity\focused-babbage\failing_classes.txt"
    if os.path.exists(failing_txt_path):
        with open(failing_txt_path, 'r') as f:
            failing_names = {line.strip().lower() for line in f if line.strip()}
        print(f"Loaded {len(failing_names)} custom oversampling targets from failing_classes.txt")
        
    for name, path in template_paths.items():
        try:
            with Image.open(path) as img:
                rgba = img.convert('RGBA')
                bw = 300
                bh = int((bw / rgba.size[0]) * rgba.size[1])
                resized = rgba.resize((bw, bh), Image.Resampling.LANCZOS)
                templates.append(resized)
                
                lower_name = name.lower()
                # 1. Check for custom failing classes
                if lower_name in failing_names:
                    failing_templates.append(resized)
                    print(f"Failing class target loaded: {name}")
                # 2. Check for Qatar or Bahrain templates
                if lower_name in ['qa', 'qa_flag', 'bh', 'bh_flag'] or 'qatar' in lower_name or 'bahrain' in lower_name:
                    qatar_templates.append(resized)
        except Exception as e:
            print(f"  Skip {path}: {e}")

    print(f"Loaded {len(templates)} templates. Found {len(qatar_templates)} Qatar/Bahrain targets and {len(failing_templates)} custom failing targets.")

    for split, count in [('train', num_train), ('val', num_val)]:
        print(f"\nGenerating {count} images for {split}...")
        for i in range(count):
            bg = random.choice(backgrounds).copy()
            # 20% negative rate to suppress false positives on rocks/runways
            is_neg = random.random() < 0.20   

            if is_neg:
                img = camera_effects(bg)
                lbl = ""
            else:
                # Priority oversampling feedback loop:
                # If we have custom failing templates, draw from them with 25% probability.
                # If we have Qatar/Bahrain templates, draw from them with 15% probability.
                # Otherwise, draw from standard templates.
                r_val = random.random()
                if r_val < 0.25 and failing_templates:
                    flag = random.choice(failing_templates)
                elif r_val < 0.40 and qatar_templates:
                    flag = random.choice(qatar_templates)
                else:
                    flag = random.choice(templates)
                    
                img, lbl = overlay_flag(bg, flag)
                img = camera_effects(img)

            fname = f"flag_synth_{split}_{i:05d}"
            img.save(os.path.join(base, 'images', split, f"{fname}.jpg"),
                     'JPEG', quality=random.randint(75, 95))
            with open(os.path.join(base, 'labels', split, f"{fname}.txt"), 'w') as f:
                if lbl:
                    f.write(lbl + "\n")

            if (i+1) % 500 == 0 or i+1 == count:
                print(f"  {i+1}/{count}")

    # Write 1-class yaml
    yaml = f"""path: {base.replace(chr(92), '/')}
train: images/train
val: images/val

names:
  0: flag
"""
    with open(os.path.join(base, 'dataset_1class.yaml'), 'w') as f:
        f.write(yaml)
    print(f"\nDataset ready! YAML: {os.path.join(base, 'dataset_1class.yaml')}")


if __name__ == '__main__':
    generate_dataset()
