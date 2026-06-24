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

def loss_function_l2(params, data_points, W, H, f, phi, gps_t, gps_n, gps_e, gps_d, headings, dt_offset):
    N_f, E_f, alt_g = params
    x_0 = W / 2.0
    y_0 = H / 2.0
    
    total_loss = 0
    for dp in data_points:
        t_sec = dp['t_sec'] + dt_offset
        
        # Interpolate drone state with shifted timestamp
        n_d = np.interp(t_sec, gps_t, gps_n)
        e_d = np.interp(t_sec, gps_t, gps_e)
        d_d = np.interp(t_sec, gps_t, gps_d)
        alt_d = dp['alt_0'] - d_d
        yaw_d = np.interp(t_sec, gps_t, headings)
        
        theta = yaw_d + phi
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        h = alt_d - alt_g
        if h <= 0.1:
            return 1e12
            
        dy = N_f - n_d
        dx = E_f - e_d
        
        X_c = dx * cos_t - dy * sin_t
        Y_c = -dx * sin_t - dy * cos_t
        
        x_proj = x_0 + f * X_c / h
        y_proj = y_0 + f * Y_c / h
        
        dist = np.sqrt((x_proj - dp['x_pixel'])**2 + (y_proj - dp['y_pixel'])**2)
        total_loss += dist**2
        
    return total_loss

def main():
    video_name = 'GL014208.LRV'
    gps_json_path = r'C:\Users\mido\Documents\antigravity\focused-babbage\meta_data\GX014208_1_GPS9.json'
    
    detections = load_detections()[video_name]
    gps_data = sorted(load_gps(gps_json_path), key=lambda x: x['cts'])
    
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
    
    last_h = 0.0
    for idx in range(n_samples):
        if headings[idx] == 0.0:
            headings[idx] = last_h
        else:
            last_h = headings[idx]
            
    cap = cv2.VideoCapture(r'C:\Users\mido\Documents\antigravity\focused-babbage\test_flight\\' + video_name)
    W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    
    f = 3000.0 * (W / 3840.0)
    
    # Filter detections to ONLY keep the moving phase (t between 100.0s and 140.0s)
    moving_detections = [d for d in detections if 100.0 <= d['t_sec'] <= 140.0]
    
    data_points = []
    for d in moving_detections:
        data_points.append({
            'frame': d['frame'],
            't_sec': d['t_sec'],
            'alt_0': alt0,
            'x_pixel': d['x'],
            'y_pixel': d['y']
        })
        
    dt_offsets = np.linspace(-1.0, 1.0, 41) # 0.05s steps
    phis = np.linspace(-np.pi, np.pi, 73)
    
    best_overall_rmse = 1e15
    best_config = None
    
    print("Sweeping time offsets (dt_offset) and camera yaw offsets (phi)...")
    for dt in dt_offsets:
        for phi_rad in phis:
            best_loss = 1e15
            best_alt_g = None
            
            # Fast grid search for alt_g
            for alt_g_val in np.linspace(alt0 - 150.0, alt0 - 20.0, 15):
                A_rows = []
                B_rows = []
                for dp in data_points:
                    t_sec = dp['t_sec'] + dt
                    n_d = np.interp(t_sec, gps_t, gps_n)
                    e_d = np.interp(t_sec, gps_t, gps_e)
                    d_d = np.interp(t_sec, gps_t, gps_d)
                    alt_d = alt0 - d_d
                    yaw_d = np.interp(t_sec, gps_t, headings)
                    
                    h = alt_d - alt_g_val
                    if h <= 0.1:
                        continue
                    theta = yaw_d + phi_rad
                    cos_t = np.cos(theta)
                    sin_t = np.sin(theta)
                    
                    A_rows.append([cos_t, -sin_t])
                    B_rows.append((dp['x_pixel'] - W/2.0) * h / f + e_d * cos_t - n_d * sin_t)
                    A_rows.append([-sin_t, -cos_t])
                    B_rows.append((dp['y_pixel'] - H/2.0) * h / f - e_d * sin_t - n_d * cos_t)
                    
                if len(A_rows) < 4:
                    continue
                xy_f, _, _, _ = np.linalg.lstsq(np.array(A_rows), np.array(B_rows), rcond=None)
                loss = loss_function_l2([xy_f[1], xy_f[0], alt_g_val], data_points, W, H, f, phi_rad, gps_t, gps_n, gps_e, gps_d, headings, dt)
                if loss < best_loss:
                    best_loss = loss
                    best_alt_g = alt_g_val
                    best_xy = xy_f
                    
            if best_alt_g is None:
                continue
                
            guess = [best_xy[1], best_xy[0], best_alt_g]
            bounds = [
                (-500.0, 500.0),
                (-500.0, 500.0),
                (alt0 - 180.0, alt0 - 10.0)
            ]
            res = opt.minimize(loss_function_l2, guess, args=(data_points, W, H, f, phi_rad, gps_t, gps_n, gps_e, gps_d, headings, dt), bounds=bounds, method='L-BFGS-B')
            if res.success:
                rmse = np.sqrt(res.fun / len(data_points))
                if rmse < best_overall_rmse:
                    best_overall_rmse = rmse
                    best_config = {
                        'dt_offset': dt,
                        'phi_deg': np.degrees(phi_rad),
                        'rmse': rmse,
                        'N_f': res.x[0],
                        'E_f': res.x[1],
                        'alt_g': res.x[2]
                    }
                    
    print("\nBest Configuration with Time Sync Calibration:")
    print(f"  Time Offset (dt):  {best_config['dt_offset']:.2f} seconds")
    print(f"  Camera Yaw Offset: {best_config['phi_deg']:.1f} degrees")
    print(f"  RMSE:              {best_config['rmse']:.2f} px")
    print(f"  Ground Altitude:   {best_config['alt_g']:.2f} m (MSL)")
    
    N_f, E_f, alt_g = best_config['N_f'], best_config['E_f'], best_config['alt_g']
    d_f = alt0 - alt_g
    flag_lat, flag_lon, flag_alt = pm.ned2geodetic(N_f, E_f, d_f, lat0, lon0, alt0)
    print(f"\nFinal Calibrated Flag Location:")
    print(f"  Latitude:  {flag_lat:.8f} N")
    print(f"  Longitude: {flag_lon:.8f} E")
    print(f"  Altitude:  {flag_alt:.2f} m (MSL)")

if __name__ == '__main__':
    main()
