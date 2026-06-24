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

def loss_function(params, data_points, W, H):
    # params: [N_f, E_f, alt_g, f, phi, x_0, y_0]
    N_f, E_f, alt_g, f, phi, x_0, y_0 = params
    
    total_loss = 0
    for dp in data_points:
        N_d = dp['n_d']
        E_d = dp['e_d']
        alt_d = dp['alt_d']
        yaw_d = dp['yaw_d']
        x_meas = dp['x_pixel']
        y_meas = dp['y_pixel']
        
        theta = yaw_d + phi
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        h = alt_d - alt_g
        if h <= 0.1:
            return 1e12
            
        dy = N_f - N_d
        dx = E_f - E_d
        
        X_c = dx * cos_t - dy * sin_t
        Y_c = -dx * sin_t - dy * cos_t
        
        x_proj = x_0 + f * X_c / h
        y_proj = y_0 + f * Y_c / h
        
        dist = np.sqrt((x_proj - x_meas)**2 + (y_proj - y_meas)**2)
        
        # Huber Loss
        delta = 20.0
        if dist < delta:
            loss = 0.5 * (dist**2)
        else:
            loss = delta * (dist - 0.5 * delta)
        total_loss += loss
        
    return total_loss

def run_fit(data_points, W, H, alt0, f_init):
    # Grid search for initialization
    best_loss = 1e15
    best_init = None
    
    ground_alts = np.linspace(alt0 - 150, alt0 - 20, 27)
    phis = np.linspace(-np.pi, np.pi, 37)
    
    for alt_g_val in ground_alts:
        for phi_val in phis:
            A_rows = []
            B_rows = []
            for dp in data_points:
                h = dp['alt_d'] - alt_g_val
                if h <= 0.1:
                    continue
                theta = dp['yaw_d'] + phi_val
                cos_t = np.cos(theta)
                sin_t = np.sin(theta)
                
                A_rows.append([cos_t, -sin_t])
                B_rows.append((dp['x_pixel'] - W/2.0) * h / f_init + dp['e_d'] * cos_t - dp['n_d'] * sin_t)
                A_rows.append([-sin_t, -cos_t])
                B_rows.append((dp['y_pixel'] - H/2.0) * h / f_init - dp['e_d'] * sin_t - dp['n_d'] * cos_t)
                
            if len(A_rows) < 4:
                continue
            xy_f, _, _, _ = np.linalg.lstsq(np.array(A_rows), np.array(B_rows), rcond=None)
            loss = loss_function([xy_f[1], xy_f[0], alt_g_val, f_init, phi_val, W/2.0, H/2.0], data_points, W, H)
            if loss < best_loss:
                best_loss = loss
                best_init = [xy_f[1], xy_f[0], alt_g_val, f_init, phi_val, W/2.0, H/2.0]
                
    if best_init is None:
        return None
        
    bounds = [
        (-500.0, 500.0),
        (-500.0, 500.0),
        (alt0 - 180.0, alt0 - 10.0),
        (0.5 * f_init, 2.0 * f_init),
        (-2.0 * np.pi, 2.0 * np.pi),
        (W/2.0 - 50.0, W/2.0 + 50.0),
        (H/2.0 - 50.0, H/2.0 + 50.0)
    ]
    
    res = opt.minimize(loss_function, best_init, args=(data_points, W, H), bounds=bounds, method='L-BFGS-B')
    return res

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
    
    # Compute headings
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
    
    # Fill in static values
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
    
    f_init = 3000.0 * (W / 3840.0)
    
    # Prepare all data points
    data_points = []
    for d in detections:
        t_sec = d['t_sec']
        n_interp = np.interp(t_sec, gps_t, gps_n)
        e_interp = np.interp(t_sec, gps_t, gps_e)
        d_interp = np.interp(t_sec, gps_t, gps_d)
        alt_interp = alt0 - d_interp
        yaw_interp = np.interp(t_sec, gps_t, headings)
        
        data_points.append({
            'frame': d['frame'],
            't_sec': t_sec,
            'n_d': n_interp,
            'e_d': e_interp,
            'alt_d': alt_interp,
            'yaw_d': yaw_interp,
            'x_pixel': d['x'],
            'y_pixel': d['y'],
            'area': d['area']
        })
        
    print(f"Total points: {len(data_points)}")
    
    # First fit
    res = run_fit(data_points, W, H, alt0, f_init)
    if not res.success:
        print("Initial fit failed.")
        return
        
    N_f, E_f, alt_g, f_val, phi, x_0, y_0 = res.x
    
    # Print reprojection errors for each frame
    print("\nReprojection Errors:")
    print(f"{'Frame':6s} | {'Time (s)':8s} | {'X_meas':6s} | {'Y_meas':6s} | {'X_proj':6s} | {'Y_proj':6s} | {'Error (px)':10s} | {'Area':6s}")
    
    filtered_points = []
    for dp in data_points:
        theta = dp['yaw_d'] + phi
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        h = dp['alt_d'] - alt_g
        dy = N_f - dp['n_d']
        dx = E_f - dp['e_d']
        X_c = dx * cos_t - dy * sin_t
        Y_c = -dx * sin_t - dy * cos_t
        x_proj = x_0 + f_val * X_c / h
        y_proj = y_0 + f_val * Y_c / h
        dist = np.sqrt((x_proj - dp['x_pixel'])**2 + (y_proj - dp['y_pixel'])**2)
        
        print(f"{dp['frame']:6d} | {dp['t_sec']:8.1f} | {dp['x_pixel']:6.1f} | {dp['y_pixel']:6.1f} | {x_proj:6.1f} | {y_proj:6.1f} | {dist:10.2f} | {dp['area']:6.1f}")
        
        # We classify as inlier if error < 40 pixels (which is very reasonable for dynamic flight)
        if dist < 40.0:
            filtered_points.append(dp)
            
    print(f"\nFiltered inliers: {len(filtered_points)} / {len(data_points)}")
    
    # Re-run fit on filtered points
    res_filtered = run_fit(filtered_points, W, H, alt0, f_init)
    if res_filtered.success:
        N_f, E_f, alt_g, f_val, phi, x_0, y_0 = res_filtered.x
        d_f = alt0 - alt_g
        flag_lat, flag_lon, flag_alt = pm.ned2geodetic(N_f, E_f, d_f, lat0, lon0, alt0)
        rmse = np.sqrt(res_filtered.fun / len(filtered_points))
        
        print(f"\n=== Refined Fit on Inliers ===")
        print(f"Optimal Local Parameters:")
        print(f"  Flag NED North:  {N_f:.4f} m")
        print(f"  Flag NED East:   {E_f:.4f} m")
        print(f"  Ground Altitude: {alt_g:.4f} m (MSL)")
        print(f"  Focal Length:    {f_val:.2f} px")
        print(f"  Camera Yaw Offset (phi): {np.degrees(phi):.2f} deg")
        print(f"  Principal Point: ({x_0:.2f}, {y_0:.2f})")
        print(f"\nFlag Geographic Location:")
        print(f"  Latitude:  {flag_lat:.8f} N")
        print(f"  Longitude: {flag_lon:.8f} E")
        print(f"  Altitude:  {flag_alt:.2f} m (MSL)")
        print(f"  RMSE:      {rmse:.2f} px")
        
        # Print remaining points errors
        print("\nRefined Reprojection Errors:")
        for dp in filtered_points:
            theta = dp['yaw_d'] + phi
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            h = dp['alt_d'] - alt_g
            dy = N_f - dp['n_d']
            dx = E_f - dp['e_d']
            X_c = dx * cos_t - dy * sin_t
            Y_c = -dx * sin_t - dy * cos_t
            x_proj = x_0 + f_val * X_c / h
            y_proj = y_0 + f_val * Y_c / h
            dist = np.sqrt((x_proj - dp['x_pixel'])**2 + (y_proj - dp['y_pixel'])**2)
            print(f"  Frame {dp['frame']:5d} ({dp['t_sec']:5.1f}s): Error = {dist:6.2f} px")

if __name__ == '__main__':
    main()
