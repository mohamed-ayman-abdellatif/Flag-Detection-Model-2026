import os
import sys
import cv2
import torch
import numpy as np
from ultralytics import YOLO

class TemporalFilter:
    def __init__(self, min_frames=5, max_lost_frames=3, iou_threshold=0.15):
        self.min_frames = min_frames
        self.max_lost_frames = max_lost_frames
        self.iou_threshold = iou_threshold
        self.tracks = []  # List of active tracks

    def update(self, new_boxes):
        # new_boxes is a list of tuples (box, conf) where box is [x1, y1, x2, y2]
        matched_track_indices = set()
        matched_new_indices = set()
        
        # 1. Match new detections with existing tracks based on IoU
        for new_idx, (new_box, new_conf) in enumerate(new_boxes):
            best_iou = -1
            best_track_idx = -1
            for track_idx, track in enumerate(self.tracks):
                if track_idx in matched_track_indices:
                    continue
                iou = self.get_iou(new_box, track['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_track_idx = track_idx
            
            if best_iou >= self.iou_threshold:
                # Update matched track
                track = self.tracks[best_track_idx]
                track['box'] = new_box
                track['conf'] = new_conf
                track['frames_active'] += 1
                track['frames_since_seen'] = 0
                matched_track_indices.add(best_track_idx)
                matched_new_indices.add(new_idx)
                
        # 2. Update unmatched existing tracks
        for track_idx, track in enumerate(self.tracks):
            if track_idx not in matched_track_indices:
                track['frames_since_seen'] += 1
                
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t['frames_since_seen'] <= self.max_lost_frames]
        
        # 3. Create new tracks for unmatched detections
        for new_idx, (new_box, new_conf) in enumerate(new_boxes):
            if new_idx not in matched_new_indices:
                self.tracks.append({
                    'box': new_box,
                    'conf': new_conf,
                    'frames_active': 1,
                    'frames_since_seen': 0
                })
                
        # 4. Return stable tracks (seen for at least min_frames, and not lost for more than 1 frame)
        stable_detections = []
        for track in self.tracks:
            if track['frames_active'] >= self.min_frames and track['frames_since_seen'] <= 1:
                stable_detections.append((track['box'], track['conf']))
        return stable_detections

    @staticmethod
    def get_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0

def main():
    weights_path = "yolo26s_flag_best.pt"
    
    # Check if weights exist
    if not os.path.exists(weights_path):
        print(f"Error: Trained model weights not found at '{weights_path}'")
        print("Please train the model first or place the weights in the workspace root.")
        return
        
    # Get video path from command line arguments, or use a default one
    default_video = r"test_flight\GX014209.MP4"
    video_path = sys.argv[1] if len(sys.argv) > 1 else default_video
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        print("Please provide a valid path to a video file. Example:")
        print(f"  python detect_flag_video.py path\\to\\your\\video.mp4")
        return
        
    print(f"=== Running YOLO26 Flag Detector on Video (Robust Mode) ===")
    print(f"Model Weights: {weights_path}")
    print(f"Video File:    {video_path}")
    
    # Load model
    print("Loading model...")
    model = YOLO(weights_path)
    
    # Check if GPU is available
    device_val = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Device:        {device_val}")
    
    # Open input video
    print("Opening video file...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: OpenCV could not open video file '{video_path}'")
        return
        
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps <= 0:
        fps = 30.0
    if total_frames <= 0:
        total_frames = 1
        
    print(f"Video resolution: {width}x{height}")
    print(f"FPS:              {fps}")
    print(f"Total Frames:     {total_frames}")
    
    # Prepare output path
    output_dir = os.path.join(os.path.dirname(os.path.abspath(video_path)) if os.path.dirname(video_path) else ".", "annotated_runs")
    os.makedirs(output_dir, exist_ok=True)
    video_basename = os.path.basename(video_path)
    video_name, video_ext = os.path.splitext(video_basename)
    # Output file path
    output_path = os.path.join(output_dir, f"{video_name}_annotated.mp4")
    
    print(f"Output File:      {output_path}")
    print("\nStarting video processing. This may take some time...")
    
    # Open VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    skipped_count = 0
    temp_filter = TemporalFilter(min_frames=5, max_lost_frames=3, iou_threshold=0.15)
    
    try:
        while True:
            # We wrap the read in a try-except to handle any decodification or OpenCV internal errors
            try:
                success, frame = cap.read()
                if not success:
                    # If read is unsuccessful, it might be the end of the video
                    if frame_idx >= total_frames - 5:
                        break
                    print(f"\n[Warning] Failed to read frame {frame_idx}. Attempting to continue...")
                    skipped_count += 1
                    frame_idx += 1
                    continue
            except Exception as read_err:
                print(f"\n[Error] Exception occurred reading frame {frame_idx}: {read_err}")
                skipped_count += 1
                frame_idx += 1
                continue
                
            # Run prediction on the frame
            try:
                results = model.predict(
                    source=frame,
                    conf=0.15,
                    imgsz=640,
                    device=device_val,
                    verbose=False
                )
                
                # Extract predicted boxes and confidences
                pred_boxes = []
                for box in results[0].boxes:
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    conf = float(box.conf[0])
                    pred_boxes.append((xyxy, conf))
                    
                # Apply temporal filter to get stable detections
                stable_boxes = temp_filter.update(pred_boxes)
                
                # Draw stable boxes onto a copy of the frame
                annotated_frame = frame.copy()
                for box, conf in stable_boxes:
                    x1, y1, x2, y2 = [int(c) for c in box]
                    # Draw a nice green bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    label = f"flag {conf:.2f}"
                    # Label text background
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated_frame, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Write annotated frame to output video
                writer.write(annotated_frame)
            except Exception as pred_err:
                print(f"\n[Error] Exception occurred during prediction on frame {frame_idx}: {pred_err}")
                # Write the original unannotated frame so the video doesn't desync
                writer.write(frame)
                skipped_count += 1
                
            frame_idx += 1
            if frame_idx % 50 == 0 or frame_idx == total_frames:
                progress = (frame_idx / total_frames) * 100
                print(f"Processed {frame_idx}/{total_frames} frames ({progress:.1f}%) | Skipped: {skipped_count}", end='\r')
                sys.stdout.flush()
                
    finally:
        # Release resources
        cap.release()
        writer.release()
        print("\n\n=== Video Processing Complete! ===")
        print(f"Total processed frames: {frame_idx - skipped_count}")
        print(f"Total skipped frames:   {skipped_count}")
        print(f"Annotated video saved successfully to: {output_path}")

if __name__ == '__main__':
    main()
