import csv
import numpy as np
from scipy.spatial.transform import Rotation as R

def load_csv(path):
    data = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({k: float(v) for k, v in row.items()})
    return data

def main():
    cori_data = load_csv('GX014209_CORI.csv')
    
    # We will check specific detection frames:
    # Frame 840 (28.0s)
    # Frame 900 (30.0s)
    # Frame 960 (32.0s)
    # Frame 1230 (41.0s)
    # Frame 2460 (82.1s)
    # Frame 6720 (224.2s)
    # Frame 7530 (251.3s)
    # Frame 7650 (255.3s)
    
    frames = [840, 900, 960, 1230, 2460, 6720, 7530, 7650]
    for frame in frames:
        t_sec = frame / 29.97
        t_ms = t_sec * 1000.0
        
        # Find closest CORI
        closest = min(cori_data, key=lambda x: abs(x['cts_ms'] - t_ms))
        
        q = [closest['qx'], closest['qy'], closest['qz'], closest['qw']]
        rot = R.from_quat(q)
        euler = rot.as_euler('zyx', degrees=True) # yaw, pitch, roll
        
        print(f"Frame {frame:4d} ({t_sec:5.1f}s): CORI Yaw = {euler[0]:6.1f} deg, Pitch = {euler[1]:5.1f} deg, Roll = {euler[2]:5.1f} deg")

if __name__ == '__main__':
    main()
