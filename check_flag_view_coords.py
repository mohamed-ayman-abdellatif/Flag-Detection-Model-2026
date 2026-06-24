import json
import os

def parse_logs():
    log_path = r'C:\Users\mido\.gemini\antigravity\brain\7c583763-3713-4256-b5ad-dfa91099fc03\.system_generated\tasks\task-287.log'
    with open(log_path, 'r') as f:
        lines = f.readlines()

    current_video = None
    detections = {}

    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith("Sequential Scanning:"):
            video_path = line.split("Sequential Scanning:")[1].split("(")[0].strip()
            video_name = os.path.basename(video_path)
            current_video = video_name
            detections[current_video] = []
            
        elif line.startswith("Flag detected at frame"):
            if current_video is None:
                continue
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
    
    gps_map = {
        'GL014208.LRV': r'C:\Users\mido\Documents\antigravity\focused-babbage\meta_data\GX014208_1_GPS9.json',
        'GX014209.MP4': r'C:\Users\mido\Documents\antigravity\focused-babbage\meta_data\GX014209_1_GPS9.json'
    }
    
    for vid, json_path in gps_map.items():
        if vid not in detections:
            continue
        print(f"\nAll detections for {vid}:")
        gps_data = load_gps_json(json_path)
        dets = detections[vid]
        
        for det in dets:
            t_ms = det['t_sec'] * 1000.0
            closest_gps = min(gps_data, key=lambda x: abs(x['cts'] - t_ms))
            print(f"  Frame {det['frame']:5d} ({det['t_sec']:5.1f}s): Center=({det['x']:6.1f}, {det['y']:6.1f}), Area={det['area']:6.1f} | Drone GPS: Lat={closest_gps['lat']:.8f}, Lon={closest_gps['lon']:.8f}, Alt={closest_gps['alt']:.2f}m")

if __name__ == '__main__':
    main()
