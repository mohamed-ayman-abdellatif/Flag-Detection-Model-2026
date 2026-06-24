import csv
import numpy as np
import pymap3d as pm
import scipy.optimize as opt

def load_data():
    telemetry = {}
    with open('drone_telemetry.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['lat'] and row['lon'] and row['alt']:
                telemetry[row['frame']] = {
                    'lat': float(row['lat']),
                    'lon': float(row['lon']),
                    'alt': float(row['alt'])
                }
                
    detections = {}
    with open('flag_detections.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['detected'] == '1':
                detections[row['frame']] = {
                    'x': float(row['x']),
                    'y': float(row['y']),
                    'area': float(row['area'])
                }
    return telemetry, detections

def filter_consecutive(detections):
    # Detections keys are sorted frame names, e.g. frame_0000.jpg, frame_0001.jpg
    sorted_frames = sorted(detections.keys())
    
    # Parse frame indices
    def get_index(f):
        return int(f.split('_')[1].split('.')[0])
        
    filtered = {}
    n = len(sorted_frames)
    
    for i in range(n):
        f = sorted_frames[i]
        idx = get_index(f)
        
        # Check if there is a neighbor within 2 frames (either before or after)
        has_prev = False
        has_next = False
        
        # Look backward
        for j in range(i-1, -1, -1):
            prev_idx = get_index(sorted_frames[j])
            if idx - prev_idx <= 2:
                has_prev = True
                break
            elif idx - prev_idx > 2:
                break
                
        # Look forward
        for j in range(i+1, n):
            next_idx = get_index(sorted_frames[j])
            if next_idx - idx <= 2:
                has_next = True
                break
            elif next_idx - idx > 2:
                break
                
        # Keep if it has a neighbor within 2 frames
        # To make it even more robust, it must have BOTH a prev and a next (part of a sequence of size >= 3), 
        # or be next to a close sequence.
        if has_prev or has_next:
            filtered[f] = detections[f]
            
    return filtered

def compute_headings(telemetry):
    sorted_frames = sorted(telemetry.keys())
    lat0 = telemetry[sorted_frames[0]]['lat']
    lon0 = telemetry[sorted_frames[0]]['lon']
    alt0 = telemetry[sorted_frames[0]]['alt']
    
    ned_coords = {}
    for frame in sorted_frames:
        t = telemetry[frame]
        n, e, d = pm.geodetic2ned(t['lat'], t['lon'], t['alt'], lat0, lon0, alt0)
        ned_coords[frame] = (n, e, d)
        
    headings = {}
    n_frames = len(sorted_frames)
    
    for idx, frame in enumerate(sorted_frames):
        start_idx = max(0, idx - 2)
        end_idx = min(n_frames - 1, idx + 2)
        n_start, e_start, _ = ned_coords[sorted_frames[start_idx]]
        n_end, e_end, _ = ned_coords[sorted_frames[end_idx]]
        dn = n_end - n_start
        de = e_end - e_start
        if np.sqrt(dn**2 + de**2) > 0.05:
            heading = np.atan2(de, dn)
        else:
            heading = 0.0
        headings[frame] = heading
        
    last_heading = 0.0
    for frame in sorted_frames:
        if headings[frame] == 0.0:
            headings[frame] = last_heading
        else:
            last_heading = headings[frame]
            
    return ned_coords, headings, lat0, lon0, alt0

def main():
    telemetry, detections = load_data()
    filtered_detections = filter_consecutive(detections)
    
    print(f"Original detections: {len(detections)}")
    print(f"Filtered (consecutive) detections: {len(filtered_detections)}")
    
    # Save the filtered detections to check them
    for f in sorted(filtered_detections.keys())[:10]:
        print(f"Keep: {f} -> {filtered_detections[f]}")
        
if __name__ == '__main__':
    main()
