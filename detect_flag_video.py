import os
import sys
import cv2
import torch
from ultralytics import YOLO

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
                
                # Plot predictions back onto the frame
                annotated_frame = results[0].plot()
                
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
