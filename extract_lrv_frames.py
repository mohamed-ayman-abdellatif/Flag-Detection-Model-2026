import cv2
import os

def extract_frame(video_path, frame_idx, output_path):
    print(f"Extracting frame {frame_idx} from {video_path}...")
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if ret:
        small = cv2.resize(frame, (960, 540))
        cv2.imwrite(output_path, small)
        print(f"Saved to {output_path}")
    else:
        print(f"Failed to extract frame {frame_idx}")
    cap.release()

def main():
    video_path = r'C:\Users\mido\Documents\antigravity\focused-babbage\test_flight\GL014208.LRV'
    art_dir = r'C:\Users\mido\.gemini\antigravity\brain\7c583763-3713-4256-b5ad-dfa91099fc03'
    
    extract_frame(video_path, 900, os.path.join(art_dir, 'lrv_frame_0900_small.jpg'))
    extract_frame(video_path, 3330, os.path.join(art_dir, 'lrv_frame_3330_small.jpg'))

if __name__ == '__main__':
    main()
