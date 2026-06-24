import json
import os
import numpy as np

def parse_logs():
    log_path = r'C:\Users\mido\.gemini\antigravity\brain\7c583763-3713-4256-b5ad-dfa91099fc03\.system_generated\tasks\task-287.log'
    if not os.path.exists(log_path):
        print("Log file not found!")
        return {}, {}

    with open(log_path, 'r') as f:
        lines = f.readlines()

    current_video = None
    detections = {}

    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith("Sequential Scanning:"):
            # Sequential Scanning: C:\Users\mido\Documents\antigravity\focused-babbage\test_flight\GL014208.LRV (15747 frames)...
            # Extract video path
            video_path = line.split("Sequential Scanning:")[1].split("(")[0].strip()
            video_name = os.path.basename(video_path)
            current_video = video_name
            detections[current_video] = []
            print(f"Parsing detections for video: {current_video}")
            
        elif line.startswith("Flag detected at frame"):
            if current_video is None:
                continue
            # e.g., Flag detected at frame 900 (30.0s): Center=(25.0, 246.2), Area=37.0
            import re
            match = re.search(r'frame (\d+) \(([\d\.]+)s\): Center=\(([\d\.-]+), ([\d\.-]+)\), Area=([\d\.-]+)', line)
            if match:
                frame = int(match.group(1))
                time_sec = float(match.group(2))
                x = float(match.group(3))
                y = float(match.group(4))
                area = float(match.group(5))
                
                detections[current_video].append({
                    'frame': frame,
                    't_sec': time_sec,
                    'x': x,
                    'y': y,
                    'area': area
                })
            
        elif line.startswith("Finished scanning"):
            current_video = None

    for vid, dets in detections.items():
        print(f"Video {vid}: found {len(dets)} detections")
        
    return detections

def load_gps_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    streams = data.get("1", {}).get("streams", {})
    if "GPS9" not in streams:
        for k in data.keys():
            streams = data[k].get("streams", {})
            if "GPS9" in streams:
                break
    samples = streams["GPS9"]["samples"]
    gps_data = []
    for s in samples:
        val = s["value"]
        gps_data.append({
            'cts': s['cts'], # ms
            'lat': val[0],
            'lon': val[1],
            'alt': val[2],
            'fix': val[8]
        })
    return gps_data

def main():
    detections = parse_logs()
    
    # Let's map videos to GPS json files
    # GL014208.LRV -> GX014208_1_GPS9.json
    # GX014209.MP4 -> GX014209_1_GPS9.json
    gps_map = {
        'GL014208.LRV': r'C:\Users\mido\Documents\antigravity\focused-babbage\meta_data\GX014208_1_GPS9.json',
        'GX014209.MP4': r'C:\Users\mido\Documents\antigravity\focused-babbage\meta_data\GX014209_1_GPS9.json'
    }
    
    for vid, json_path in gps_map.items():
        if vid not in detections:
            continue
        print(f"\nMatching detections with GPS for {vid}...")
        gps_data = load_gps_json(json_path)
        dets = detections[vid]
        
        # print first few matches
        for det in dets[:5]:
            t_ms = det['t_sec'] * 1000.0
            closest_gps = min(gps_data, key=lambda x: abs(x['cts'] - t_ms))
            time_diff = (closest_gps['cts'] - t_ms) / 1000.0
            print(f"  Frame {det['frame']} ({det['t_sec']:.1f}s): Center=({det['x']:.1f}, {det['y']:.1f}), Area={det['area']:.1f} | GPS: Lat={closest_gps['lat']:.8f}, Lon={closest_gps['lon']:.8f}, Alt={closest_gps['alt']:.2f}m (dt={time_diff:.3f}s)")

if __name__ == '__main__':
    main()
