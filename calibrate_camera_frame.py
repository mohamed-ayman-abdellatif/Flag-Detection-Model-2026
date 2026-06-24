import csv
import json
import os
import numpy as np
import scipy.optimize as opt
import pymap3d as pm
import cv2
from scipy.spatial.transform import Rotation as R

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
    # Generate all 24 right-handed rotation/permutation matrices
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
            # Check if determinant is +1 (right-handed rotation)
            if np.isclose(np.linalg.det(M), 1.0):
                possible_R0s.append(M)
    return possible_R0s

def loss_function(params, data_points, W, H, f, R0, use_transpose):
    N_f, E_f, alt_g, yaw_startup = params
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
        
        # Huber Loss
        delta = 20.0
        if dist < delta:
            loss = 0.5 * (dist**2)
        else:
            loss = delta * (dist - 0.5 * delta)
        total_loss += loss
        
    return total_loss

def main():
    video_name = 'GL014208.LRV'
    gps_json_path = r'meta_data\GX014208_1_GPS9.json'
    cori_csv_path = 'GL014208_CORI.csv'
    
    detections = load_detections()[video_name]
    gps_data = sorted(load_gps(gps_json_path), key=lambda x: x['cts'])
    cori_data = sorted(load_cori(cori_csv_path), key=lambda x: x['cts_ms'])
    
    lat0, lon0, alt0 = gps_data[0]['lat'], gps_data[0]['lon'], gps_data[0]['alt']
    
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
    
    cori_t = np.array([c['cts_ms'] / 1000.0 for c in cori_data])
    cori_q = np.array([c['q'] for c in cori_data])
    
    cap = cv2.VideoCapture(r'test_flight\\' + video_name)
    W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    
    f = 3000.0 * (W / 3840.0)
    
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
    print(f"Total possible alignments (R0): {len(possible_R0s)}")
    
    best_overall_loss = 1e15
    best_config = None
    
    # Try all alignments
    for alignment_idx, R0 in enumerate(possible_R0s):
        for use_transpose in [False, True]:
            # Run optimization from multiple random starts
            best_local_loss = 1e15
            best_local_res = None
            
            for _ in range(5):
                init_N = np.random.uniform(-100.0, 100.0)
                init_E = np.random.uniform(-100.0, 100.0)
                init_alt_g = np.random.uniform(alt0 - 150.0, alt0 - 20.0)
                init_ys = np.random.uniform(-np.pi, np.pi)
                
                guess = [init_N, init_E, init_alt_g, init_ys]
                bounds = [
                    (-500.0, 500.0),
                    (-500.0, 500.0),
                    (alt0 - 180.0, alt0 - 10.0),
                    (-2.0 * np.pi, 2.0 * np.pi)
                ]
                
                res = opt.minimize(loss_function, guess, args=(data_points, W, H, f, R0, use_transpose), bounds=bounds, method='L-BFGS-B')
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
                        'params': best_local_res.x
                    }
                    
    print("\nBest Configuration Found:")
    print(f"  Alignment Index: {best_config['alignment_idx']}")
    print(f"  Use Transpose:   {best_config['use_transpose']}")
    print(f"  RMSE:            {best_config['rmse']:.2f} px")
    
    N_f, E_f, alt_g, yaw_startup = best_config['params']
    d_f = alt0 - alt_g
    flag_lat, flag_lon, flag_alt = pm.ned2geodetic(N_f, E_f, d_f, lat0, lon0, alt0)
    print(f"\nFlag Geographic Location:")
    print(f"  Latitude:  {flag_lat:.8f} N")
    print(f"  Longitude: {flag_lon:.8f} E")
    print(f"  Altitude:  {flag_alt:.2f} m (MSL)")
    
    # Print errors for each frame with the best configuration
    print("\nIndividual Frame Errors:")
    R0 = best_config['R0']
    use_transpose = best_config['use_transpose']
    cos_ys = np.cos(yaw_startup)
    sin_ys = np.sin(yaw_startup)
    R_z = np.array([
        [cos_ys, -sin_ys, 0],
        [sin_ys, cos_ys, 0],
        [0, 0, 1]
    ])
    R_world_to_initial = R_z @ R0
    x_0 = W / 2.0
    y_0 = H / 2.0
    
    inliers = []
    for dp in data_points:
        R_rel = R.from_quat(dp['q_rel']).as_matrix()
        if use_transpose:
            R_rel = R_rel.T
        R_cam_to_ned = R_world_to_initial @ R_rel
        R_ned_to_cam = R_cam_to_ned.T
        v_NED = np.array([N_f - dp['n_d'], E_f - dp['e_d'], dp['alt_d'] - alt_g])
        v_cam = R_ned_to_cam @ v_NED
        x_proj = x_0 + f * v_cam[0] / v_cam[2]
        y_proj = y_0 + f * v_cam[1] / v_cam[2]
        dist = np.sqrt((x_proj - dp['x_pixel'])**2 + (y_proj - dp['y_pixel'])**2)
        print(f"  Frame {dp['frame']:5d} ({dp['t_sec']:5.1f}s): Error = {dist:6.2f} px")
        if dist < 30.0:
            inliers.append(dp)
            
    print(f"\nInliers count (err < 30 px): {len(inliers)} / {len(data_points)}")

if __name__ == '__main__':
    main()
