import csv
import json
import numpy as np
import pymap3d as pm
from scipy.spatial.transform import Rotation as R

def load_csv(path):
    data = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({k: float(v) for k, v in row.items()})
    return data

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

def main():
    cori_data = load_csv('GL014208_CORI.csv')
    gps_data = load_gps(r'meta_data\GX014208_1_GPS9.json')
    
    print(f"Loaded {len(cori_data)} CORI samples")
    print(f"Loaded {len(gps_data)} GPS samples")
    
    # Let's align them by time
    # Compute headings from GPS
    lat0, lon0, alt0 = gps_data[0]['lat'], gps_data[0]['lon'], gps_data[0]['alt']
    
    gps_n, gps_e, gps_t = [], [], []
    for g in gps_data:
        n, e, d = pm.geodetic2ned(g['lat'], g['lon'], g['alt'], lat0, lon0, alt0)
        gps_n.append(n)
        gps_e.append(e)
        gps_t.append(g['cts'])
        
    gps_n = np.array(gps_n)
    gps_e = np.array(gps_e)
    gps_t = np.array(gps_t)
    
    # Calculate velocity heading
    headings = []
    n_samples = len(gps_data)
    for idx in range(n_samples):
        start_idx = max(0, idx - 10)
        end_idx = min(n_samples - 1, idx + 10)
        dn = gps_n[end_idx] - gps_n[start_idx]
        de = gps_e[end_idx] - gps_e[start_idx]
        if np.sqrt(dn**2 + de**2) > 0.1:
            headings.append(np.atan2(de, dn))
        else:
            headings.append(0.0)
    headings = np.array(headings)
    
    # Match and compare
    matched = []
    for c in cori_data[::100]: # check every 100th sample
        t_ms = c['cts_ms']
        # Find closest GPS
        idx = np.argmin(np.abs(gps_t - t_ms))
        g = gps_data[idx]
        n = gps_n[idx]
        e = gps_e[idx]
        h = headings[idx]
        
        # Convert quaternion to rotation
        q = [c['qx'], c['qy'], c['qz'], c['qw']] # scipy uses [x,y,z,w]
        rot = R.from_quat(q)
        euler = rot.as_euler('zyx', degrees=True) # yaw, pitch, roll
        
        matched.append({
            't': t_ms / 1000.0,
            'gps_heading_deg': np.degrees(h),
            'cori_yaw_deg': euler[0],
            'cori_pitch_deg': euler[1],
            'cori_roll_deg': euler[2]
        })
        
    print("\nCompare GPS Heading (calculated from trajectory) and CORI Yaw (from IMU relative rotation):")
    print(f"{'Time (s)':8s} | {'GPS Heading (deg)':18s} | {'CORI Yaw (deg)':15s} | {'Diff (deg)':10s} | {'Pitch (deg)':12s} | {'Roll (deg)':10s}")
    for m in matched[:20]:
        diff = (m['gps_heading_deg'] - m['cori_yaw_deg']) % 360
        if diff > 180:
            diff -= 360
        print(f"{m['t']:8.2f} | {m['gps_heading_deg']:18.2f} | {m['cori_yaw_deg']:15.2f} | {diff:10.2f} | {m['cori_pitch_deg']:12.2f} | {m['cori_roll_deg']:10.2f}")

if __name__ == '__main__':
    main()
