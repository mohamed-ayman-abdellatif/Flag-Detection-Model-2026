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

def project_error_single_frame(theta, N_f, E_f, alt_g, dp, W, H, f):
    x_0 = W / 2.0
    y_0 = H / 2.0
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    h = dp['alt_d'] - alt_g
    dy = N_f - dp['n_d']
    dx = E_f - dp['e_d']
    X_c = dx * cos_t - dy * sin_t
    Y_c = -dx * sin_t - dy * cos_t
    x_proj = x_0 + f * X_c / h
    y_proj = y_0 + f * Y_c / h
    return (x_proj - dp['x_pixel'])**2 + (y_proj - dp['y_pixel'])**2

def main():
    video_name = 'GX014209.MP4'
    gps_json_path = r'C:\Users\mido\Documents\antigravity\focused-babbage\meta_data\GX014209_1_GPS9.json'
    
    # Calibrated flag coordinates from GL014208:
    flag_lat = 29.81838780
    flag_lon = 30.82967511
    flag_alt = 173.99
    
    detections = load_detections()[video_name]
    gps_data = sorted(load_gps(gps_json_path), key=lambda x: x['cts'])
    
    # Use origin of NED alignment (GL014208 first GPS)
    gps_json_dyn = r'C:\Users\mido\Documents\antigravity\focused-babbage\meta_data\GX014208_1_GPS9.json'
    gps_data_dyn = sorted(load_gps(gps_json_dyn), key=lambda x: x['cts'])
    lat0 = gps_data_dyn[0]['lat']
    lon0 = gps_data_dyn[0]['lon']
    alt0 = gps_data_dyn[0]['alt']
    
    N_f, E_f, _ = pm.geodetic2ned(flag_lat, flag_lon, flag_alt, lat0, lon0, alt0)
    
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
    
    print("\nFrame-by-Frame Yaw Optimization on GX014209.MP4:")
    print(f"{'Frame':6s} | {'Time (s)':8s} | {'X_pixel':8s} | {'Y_pixel':8s} | {'Opt Yaw (deg)':15s} | {'Error (px)':10s}")
    
    sq_errors = []
    
    for d in detections:
        t_sec = d['t_sec']
        n_interp = np.interp(t_sec, gps_t, gps_n)
        e_interp = np.interp(t_sec, gps_t, gps_e)
        d_interp = np.interp(t_sec, gps_t, gps_d)
        alt_interp = alt0 - d_interp
        
        dp = {
            'alt_d': alt_interp,
            'n_d': n_interp,
            'e_d': e_interp,
            'x_pixel': d['x'],
            'y_pixel': d['y']
        }
        
        # Optimize yaw for this frame
        best_loss = 1e15
        best_theta = None
        for theta_val in np.linspace(-np.pi, np.pi, 361):
            loss = project_error_single_frame(theta_val, N_f, E_f, flag_alt, dp, W, H, f)
            if loss < best_loss:
                best_loss = loss
                best_theta = theta_val
                
        res = opt.minimize(project_error_single_frame, [best_theta], args=(N_f, E_f, flag_alt, dp, W, H, f), bounds=[(-2.0*np.pi, 2.0*np.pi)], method='L-BFGS-B')
        
        if res.success:
            theta_opt = res.x[0]
            err = np.sqrt(res.fun)
            sq_errors.append(res.fun)
            print(f"{d['frame']:6d} | {t_sec:8.1f} | {d['x']:8.1f} | {d['y']:8.1f} | {np.degrees(theta_opt):15.2f} | {err:10.2f}")
            
    rmse = np.sqrt(np.mean(sq_errors))
    print(f"\nOverall Frame-by-Frame RMSE: {rmse:.2f} px")

if __name__ == '__main__':
    main()
