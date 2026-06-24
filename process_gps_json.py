import json
import os
import numpy as np

def analyze_gps_file(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
        
    print(f"\nAnalyzing: {path}")
    with open(path, 'r') as f:
        data = json.load(f)
        
    # GoPro JSON structure: {"1": {"streams": {"GPS9": {"samples": [...]}}}}
    # Let's find the GPS9 stream
    streams = data.get("1", {}).get("streams", {})
    if "GPS9" not in streams:
        # Check other keys if "1" is not there
        for key in data.keys():
            streams = data[key].get("streams", {})
            if "GPS9" in streams:
                break
                
    if "GPS9" not in streams:
        print("GPS9 stream not found in JSON")
        return
        
    gps9 = streams["GPS9"]
    samples = gps9.get("samples", [])
    print(f"Number of GPS9 samples: {len(samples)}")
    if not samples:
        return
        
    lats = []
    lons = []
    alts = []
    times = []
    
    for s in samples:
        val = s["value"] # [lat, lon, alt, spd2d, spd3d, days, secs, dop, fix]
        lats.append(val[0])
        lons.append(val[1])
        alts.append(val[2])
        times.append(s["cts"])
        
    lats = np.array(lats)
    lons = np.array(lons)
    alts = np.array(alts)
    times = np.array(times)
    
    print(f"  CTS Range: {times.min():.2f}ms to {times.max():.2f}ms (Duration: {(times.max()-times.min())/1000.0:.2f}s)")
    print(f"  Latitude:  [{lats.min():.8f}, {lats.max():.8f}]")
    print(f"  Longitude: [{lons.min():.8f}, {lons.max():.8f}]")
    print(f"  Altitude:  [{alts.min():.2f}, {alts.max():.2f}] m")
    print(f"  Mean GPS Hz: {len(samples) / ((times.max() - times.min()) / 1000.0):.2f}")

if __name__ == '__main__':
    meta_dir = r'C:\Users\mido\Documents\antigravity\focused-babbage\meta_data'
    for f in sorted(os.listdir(meta_dir)):
        if f.endswith('GPS9.json'):
            analyze_gps_file(os.path.join(meta_dir, f))
