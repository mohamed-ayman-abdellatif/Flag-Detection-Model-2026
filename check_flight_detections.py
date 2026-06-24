import json
import os
import re

# Load detections from log output
log_path = r'C:\Users\mido\.gemini\antigravity\brain\7c583763-3713-4256-b5ad-dfa91099fc03\.system_generated\tasks\task-287.log'

with open(log_path, 'r') as f:
    log_content = f.read()

# Extract detections for GL014208.LRV
lrv_detections = []
lrv_section = re.search(r'Sequential Scanning:.*GL014208.LRV.*?\n(.*?)Finished scanning', log_content, re.DOTALL)
if lrv_section:
    for line in lrv_section.group(1).strip().split('\n'):
        match = re.search(r'frame (\d+) \(([\d\.]+)s\): Center=\(([\d\.]+), ([\d\.]+)\), Area=([\d\.]+)', line)
        if match:
            lrv_detections.append({
                'frame': int(match.group(1)),
                't_sec': float(match.group(2)),
                'x': float(match.group(3)),
                'y': float(match.group(4)),
                'area': float(match.group(5))
            })

# Extract detections for GX014209.MP4
mp4_detections = []
mp4_section = re.search(r'Sequential Scanning:.*GX014209.MP4.*?\n(.*?)Finished scanning', log_content, re.DOTALL)
if mp4_section:
    for line in mp4_section.group(1).strip().split('\n'):
        match = re.search(r'frame (\d+) \(([\d\.]+)s\): Center=\(([\d\.]+), ([\d\.]+)\), Area=([\d\.]+)', line)
        if match:
            mp4_detections.append({
                'frame': int(match.group(1)),
                't_sec': float(match.group(2)),
                'x': float(match.group(3)),
                'y': float(match.group(4)),
                'area': float(match.group(5))
            })

print(f"Loaded {len(lrv_detections)} detections for GL014208.LRV")
print(f"Loaded {len(mp4_detections)} detections for GX014209.MP4")

# Function to get GPS at cts
def load_gps(json_path):
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

gps_4208 = load_gps(r'C:\Users\mido\Documents\antigravity\focused-babbage\meta_data\GX014208_1_GPS9.json')
gps_4209 = load_gps(r'C:\Users\mido\Documents\antigravity\focused-babbage\meta_data\GX014209_1_GPS9.json')

# Check what the GPS coordinates are during detections for 4208
print("\nFirst 5 detections in GL014208.LRV matched with GPS9:")
for det in lrv_detections[:5]:
    t_ms = det['t_sec'] * 1000.0
    # find closest GPS
    closest_gps = min(gps_4208, key=lambda x: abs(x['cts'] - t_ms))
    print(f"  Frame {det['frame']} ({det['t_sec']:.1f}s): Det Center=({det['x']:.1f}, {det['y']:.1f}), Area={det['area']:.1f} | Closest GPS t={closest_gps['cts']/1000.0:.2f}s, Lat={closest_gps['lat']:.8f}, Lon={closest_gps['lon']:.8f}, Alt={closest_gps['alt']:.2f}m")

# Check what the GPS coordinates are during detections for 4209
print("\nFirst 5 detections in GX014209.MP4 matched with GPS9:")
for det in mp4_detections[:5]:
    t_ms = det['t_sec'] * 1000.0
    # find closest GPS
    closest_gps = min(gps_4209, key=lambda x: abs(x['cts'] - t_ms))
    print(f"  Frame {det['frame']} ({det['t_sec']:.1f}s): Det Center=({det['x']:.1f}, {det['y']:.1f}), Area={det['area']:.1f} | Closest GPS t={closest_gps['cts']/1000.0:.2f}s, Lat={closest_gps['lat']:.8f}, Lon={closest_gps['lon']:.8f}, Alt={closest_gps['alt']:.2f}m")
