import json
import os
import numpy as np
import scipy.optimize as opt
import pymap3d as pm
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

def loss_function_yaw_only(theta, N_f, E_f, alt_g, data_points, W, H, f):
    x_0 = W / 2.0
    y_0 = H / 2.0
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    total_loss = 0
    for dp in data_points:
        h = dp['alt_d'] - alt_g
        if h <= 0.1:
            return 1e12
        dy = N_f - dp['n_d']
        dx = E_f - dp['e_d']
        X_c = dx * cos_t - dy * sin_t
        Y_c = -dx * sin_t - dy * cos_t
        x_proj = x_0 + f * X_c / h
        y_proj = y_0 + f * Y_c / h
        total_loss += (x_proj - dp['x_pixel'])**2 + (y_proj - dp['y_pixel'])**2
    return total_loss

def main():
    video_name = 'GX014209.MP4'
    gps_json_path = r'C:\Users\mido\Documents\antigravity\focused-babbage\meta_data\GX014209_1_GPS9.json'
    
    # Target flag coordinates from GL014208 moving sequence fit:
    flag_lat = 29.81838780
    flag_lon = 30.82967511
    flag_alt = 173.99
    
    detections = load_detections()[video_name]
    gps_data = sorted(load_gps(gps_json_path), key=lambda x: x['cts'])
    
    # Use the same origin for NED alignment (GL014208 first GPS)
    gps_json_dyn = r'C:\Users\mido\Documents\antigravity\focused-babbage\meta_data\GX014208_1_GPS9.json'
    gps_data_dyn = sorted(load_gps(gps_json_dyn), key=lambda x: x['cts'])
    lat0 = gps_data_dyn[0]['lat']
    lon0 = gps_data_dyn[0]['lon']
    alt0 = gps_data_dyn[0]['alt']
    
    # Calculate NED coordinates for flag
    d_f = alt0 - flag_alt
    N_f, E_f, _ = pm.geodetic2ned(flag_lat, flag_lon, flag_alt, lat0, lon0, alt0)
    print(f"Flag Local NED North: {N_f:.4f} m, East: {E_f:.4f} m")
    
    # Prepare stationary data points
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
    
    cap = cv2.VideoCapture(r'C:\Users\mido\Documents\antigravity\focused-babbage\test_flight\\' + video_name)
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
        
        data_points.append({
            'frame': d['frame'],
            't_sec': t_sec,
            'n_d': n_interp,
            'e_d': e_interp,
            'alt_d': alt_interp,
            'x_pixel': d['x'],
            'y_pixel': d['y']
        })
        
    print(f"Loaded {len(data_points)} detections from {video_name}.")
    
    # Optimize constant camera yaw theta
    best_loss = 1e15
    best_theta = None
    for theta_val in np.linspace(-np.pi, np.pi, 361):
        loss = loss_function_yaw_only(theta_val, N_f, E_f, flag_alt, data_points, W, H, f)
        if loss < best_loss:
            best_loss = loss
            best_theta = theta_val
            
    # Refine
    res = opt.minimize(loss_function_yaw_only, [best_theta], args=(N_f, E_f, flag_alt, data_points, W, H, f), bounds=[(-2.0*np.pi, 2.0*np.pi)], method='L-BFGS-B')
    if res.success:
        theta_opt = res.x[0]
        rmse = np.sqrt(res.fun / len(data_points))
        print(f"\nVerification Succeeded on Stationary Video {video_name}!")
        print(f"  Optimal Constant Camera Yaw: {np.degrees(theta_opt):.2f} degrees")
        print(f"  Reprojection RMSE on 4K video: {rmse:.2f} pixels ({rmse/W*100:.2f}% of width)")
        
        # Print projection errors for all frames
        print("\nIndividual Reprojection Errors:")
        x_0 = W / 2.0
        y_0 = H / 2.0
        cos_t = np.cos(theta_opt)
        sin_t = np.sin(theta_opt)
        for dp in data_points:
            h = dp['alt_d'] - flag_alt
            dy = N_f - dp['n_d']
            dx = E_f - dp['e_d']
            X_c = dx * cos_t - dy * sin_t
            Y_c = -dx * sin_t - dy * cos_t
            x_proj = x_0 + f * X_c / h
            y_proj = y_0 + f * Y_c / h
            dist = np.sqrt((x_proj - dp['x_pixel'])**2 + (y_proj - dp['y_pixel'])**2)
            print(f"  Frame {dp['frame']:5d} ({dp['t_sec']:5.1f}s): Error = {dist:6.2f} px")
    else:
        print("Optimization failed.")

if __name__ == '__main__':
    main()
