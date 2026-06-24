import csv
import json
import os
import numpy as np
import scipy.optimize as opt
import pymap3d as pm
from scipy.spatial.transform import Rotation as R
import cv2

def load_detections():
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

def load_cori(csv_path):
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'cts_ms': float(row['cts_ms']),
                'q': [float(row['qx']), float(row['qy']), float(row['qz']), float(row['qw'])]
            })
    return data

def get_possible_R0s():
    possible_R0s = []
    identity_permutations = [
        [0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]
    ]
    signs = [
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
    ]
    for p in identity_permutations:
        for s in signs:
            M = np.zeros((3, 3))
            M[0, p[0]] = s[0]
            M[1, p[1]] = s[1]
            M[2, p[2]] = s[2]
            if np.isclose(np.linalg.det(M), 1.0):
                possible_R0s.append(M)
    return possible_R0s

def loss_function_yaw_only(yaw_startup, data_points, W, H, f, R0, use_transpose, N_f, E_f, alt_g):
    yaw_startup = yaw_startup[0]
    x_0 = W / 2.0
    y_0 = H / 2.0
    
    cos_ys = np.cos(yaw_startup)
    sin_ys = np.sin(yaw_startup)
    R_z = np.array([
        [cos_ys, -sin_ys, 0],
        [sin_ys, cos_ys, 0],
        [0, 0, 1]
    ])
    
    R_world_to_initial = R_z @ R0
    
    total_loss = 0
    for dp in data_points:
        N_d = dp['n_d']
        E_d = dp['e_d']
        alt_d = dp['alt_d']
        x_meas = dp['x_pixel']
        y_meas = dp['y_pixel']
        
        R_rel = R.from_quat(dp['q_rel']).as_matrix()
        if use_transpose:
            R_rel = R_rel.T
            
        R_cam_to_ned = R_world_to_initial @ R_rel
        R_ned_to_cam = R_cam_to_ned.T
        
        v_NED = np.array([N_f - N_d, E_f - E_d, alt_d - alt_g])
        v_cam = R_ned_to_cam @ v_NED
        
        x_c, y_c, z_c = v_cam
        if z_c <= 0.1:
            return 1e12
            
        x_proj = x_0 + f * x_c / z_c
        y_proj = y_0 + f * y_c / z_c
        
        dist = np.sqrt((x_proj - x_meas)**2 + (y_proj - y_meas)**2)
        total_loss += dist**2
        
    return total_loss

def main():
    video_name = 'GX014209.MP4'
    gps_json_path = r'meta_data\GX014209_1_GPS9.json'
    cori_csv_path = 'GX014209_CORI.csv'
    
    # Target flag coordinates from GL014208 moving sequence fit:
    flag_lat = 29.81838780
    flag_lon = 30.82967511
    flag_alt = 173.99
    
    detections = load_detections()[video_name]
    gps_data = sorted(load_gps(gps_json_path), key=lambda x: x['cts'])
    cori_data = sorted(load_cori(cori_csv_path), key=lambda x: x['cts_ms'])
    
    # Use the same origin for NED alignment
    gps_json_dyn = r'meta_data\GX014208_1_GPS9.json'
    gps_data_dyn = sorted(load_gps(gps_json_dyn), key=lambda x: x['cts'])
    lat0 = gps_data_dyn[0]['lat']
    lon0 = gps_data_dyn[0]['lon']
    alt0 = gps_data_dyn[0]['alt']
    
    # Flag NED coordinates
    N_f, E_f, _ = pm.geodetic2ned(flag_lat, flag_lon, flag_alt, lat0, lon0, alt0)
    print(f"Flag Local NED North: {N_f:.4f} m, East: {E_f:.4f} m, Alt: {flag_alt:.2f} m")
    
    # Precompute NED for GPS
    gps_n, gps_e, gps_d, gps_t = [], [], [], []
    for g in gps_data:
        n, e, d = pm.geodetic2ned(g['lat'], g['lon'], g['alt'], lat0, lon0, alt0)
        gps_n.append(n)
        gps_e.append(e)
        gps_d.append(d)
        gps_t.append(g['cts'] / 1000.0)
    gps_n = np.array(gps_n)
    gps_e = np.array(gps_e)
    gps_d = np.array(gps_d)
    gps_t = np.array(gps_t)
    
    # Precompute CORI
    cori_t = np.array([c['cts_ms'] / 1000.0 for c in cori_data])
    cori_q = np.array([c['q'] for c in cori_data])
    
    cap = cv2.VideoCapture(r'test_flight\\' + video_name)
    W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    
    f = 3000.0
    
    data_points = []
    for d in detections:
        t_sec = d['t_sec']
        n_interp = np.interp(t_sec, gps_t, gps_n)
        e_interp = np.interp(t_sec, gps_t, gps_e)
        d_interp = np.interp(t_sec, gps_t, gps_d)
        alt_interp = alt0 - d_interp
        idx = np.argmin(np.abs(cori_t - t_sec))
        q_rel = cori_q[idx]
        
        data_points.append({
            'frame': d['frame'],
            't_sec': t_sec,
            'n_d': n_interp,
            'e_d': e_interp,
            'alt_d': alt_interp,
            'q_rel': q_rel,
            'x_pixel': d['x'],
            'y_pixel': d['y']
        })
        
    possible_R0s = get_possible_R0s()
    best_overall_loss = 1e15
    best_config = None
    
    print("\nSearching over alignments and transposes for stationary video with absolute 3D rotation...")
    for alignment_idx, R0 in enumerate(possible_R0s):
        for use_transpose in [False, True]:
            # Optimize yaw_startup only
            best_local_loss = 1e15
            best_local_yaw = None
            
            for init_ys in np.linspace(-np.pi, np.pi, 9):
                res = opt.minimize(
                    loss_function_yaw_only, [init_ys], 
                    args=(data_points, W, H, f, R0, use_transpose, N_f, E_f, flag_alt),
                    bounds=[(-2.0*np.pi, 2.0*np.pi)], method='L-BFGS-B'
                )
                if res.success and res.fun < best_local_loss:
                    best_local_loss = res.fun
                    best_local_res = res
                    
            if best_local_res is not None:
                rmse = np.sqrt(best_local_loss / len(data_points))
                if best_local_loss < best_overall_loss:
                    best_overall_loss = best_local_loss
                    best_config = {
                        'alignment_idx': alignment_idx,
                        'R0': R0,
                        'use_transpose': use_transpose,
                        'rmse': rmse,
                        'yaw_startup': best_local_res.x[0]
                    }
                    
    print("\nBest 3D Calibration Config for Stationary Video:")
    print(f"  Alignment Index: {best_config['alignment_idx']}")
    print(f"  Use Transpose:   {best_config['use_transpose']}")
    print(f"  Startup Yaw:     {np.degrees(best_config['yaw_startup']):.2f} deg")
    print(f"  RMSE:            {best_config['rmse']:.2f} px")

if __name__ == '__main__':
    main()
