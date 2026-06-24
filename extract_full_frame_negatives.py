import os
import cv2
from PIL import Image

def resize_and_letterbox(frame_bgr, target_sz=640):
    # Convert BGR (OpenCV) to RGB (PIL)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    
    w, h = img.size
    scale = target_sz / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Pad to square with neutral grey (114, 114, 114) which is standard in YOLO
    padded = Image.new("RGB", (target_sz, target_sz), (114, 114, 114))
    x_offset = (target_sz - new_w) // 2
    y_offset = (target_sz - new_h) // 2
    padded.paste(img_resized, (x_offset, y_offset))
    return padded

def extract_negatives(video_path, start_frame, end_frame, step, output_dir):
    print(f"Extracting from {os.path.basename(video_path)} (frames {start_frame}-{end_frame}, step {step})...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return 0
        
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    saved_count = 0
    
    while frame_idx <= end_frame:
        success, frame = cap.read()
        if not success:
            break
            
        if (frame_idx - start_frame) % step == 0:
            padded_img = resize_and_letterbox(frame)
            vname = os.path.splitext(os.path.basename(video_path))[0]
            out_path = os.path.join(output_dir, f"fullneg_{vname}_{frame_idx:05d}.jpg")
            padded_img.save(out_path, "JPEG", quality=85)
            saved_count += 1
            
        frame_idx += 1
        
    cap.release()
    print(f"Saved {saved_count} frames.")
    return saved_count

def main():
    video1 = r"C:\Users\mido\Documents\antigravity\focused-babbage\GX014604.MP4"
    video2 = r"C:\Users\mido\Documents\antigravity\focused-babbage\test_flight\GX014209.MP4"
    output_dir = r"C:\Users\mido\Documents\antigravity\focused-babbage\full_frame_negatives"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean output folder
    for f in os.listdir(output_dir):
        if f.endswith(".jpg"):
            try: os.remove(os.path.join(output_dir, f))
            except: pass
            
    total_saved = 0
    if os.path.exists(video1):
        # 4700 to 4950 (curbs/runways) - step 2
        total_saved += extract_negatives(video1, 4700, 4950, 2, output_dir)
        # 5550 to 5800 (rocks/uneven terrain) - step 2
        total_saved += extract_negatives(video1, 5550, 5800, 2, output_dir)
        
    if os.path.exists(video2):
        # 2000 to 2300 (runways and curbs) - step 3
        total_saved += extract_negatives(video2, 2000, 2300, 3, output_dir)
        
    print(f"\nTotal full-frame negative training images saved to {output_dir}: {total_saved}")

if __name__ == '__main__':
    main()
