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

def project_point(N_f, E_f, alt_g, phi, dp, f, W, H):
    x_0 = W / 2.0
    y_0 = H / 2.0
    theta = dp['yaw_d'] + phi
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    h = dp['alt_d'] - alt_g
    if h <= 0.1:
        return None, None
    dy = N_f - dp['n_d']
    dx = E_f - dp['e_d']
    X_c = dx * cos_t - dy * sin_t
    Y_c = -dx * sin_t - dy * cos_t
    x_proj = x_0 + f * X_c / h
    y_proj = y_0 + f * Y_c / h
    return x_proj, y_proj

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
    
    f = 3000.0 * (W / 3840.0) # 600.0
    
    # Build data points for all 43 detections
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
        
    # We will perform RANSAC
    # Since the moving sequence (t in 100 to 140s) gave a very clean fit:
    # N_f = -7.56, E_f = -56.10, AltG = 155.02, Phi = 60 deg (1.047 rad)
    # Let's verify which of the 43 points are inliers to this configuration!
    
    N_f = -7.56
    E_f = -56.10
    alt_g = 155.02
    phi = np.radians(60.0)
    
    inliers = []
    print("Evaluating all 43 detections against moving sequence calibration:")
    print(f"{'Frame':6s} | {'Time (s)':8s} | {'X_pixel':8s} | {'Y_pixel':8s} | {'X_proj':8s} | {'Y_proj':8s} | {'Error (px)':10s} | {'Status':8s}")
    for dp in data_points:
        x_proj, y_proj = project_point(N_f, E_f, alt_g, phi, dp, f, W, H)
        if x_proj is None:
            print(f"{dp['frame']:6d} | {dp['t_sec']:8.1f} | {dp['x_pixel']:8.1f} | {dp['y_pixel']:8.1f} | {'N/A':8s} | {'N/A':8s} | {'N/A':10s} | Outlier")
            continue
            
        dist = np.sqrt((x_proj - dp['x_pixel'])**2 + (y_proj - dp['y_pixel'])**2)
        status = "Inlier" if dist < 35.0 else "Outlier"
        if status == "Inlier":
            inliers.append(dp)
        print(f"{dp['frame']:6d} | {dp['t_sec']:8.1f} | {dp['x_pixel']:8.1f} | {dp['y_pixel']:8.1f} | {x_proj:8.1f} | {y_proj:8.1f} | {dist:10.2f} | {status}")
        
    print(f"\nFound {len(inliers)} inliers out of {len(data_points)} detections.")
    
    # Run optimization on inliers to refine N_f, E_f, alt_g, phi
    def loss_func(params, data):
        n_f, e_f, a_g, p = params
        x_0, y_0 = W/2.0, H/2.0
        total_loss = 0
        for dp in data:
            theta = dp['yaw_d'] + p
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            h = dp['alt_d'] - a_g
            if h <= 0.1:
                return 1e12
            dy = n_f - dp['n_d']
            dx = e_f - dp['e_d']
            X_c = dx * cos_t - dy * sin_t
            Y_c = -dx * sin_t - dy * cos_t
            x_proj = x_0 + f * X_c / h
            y_proj = y_0 + f * Y_c / h
            dist = np.sqrt((x_proj - dp['x_pixel'])**2 + (y_proj - dp['y_pixel'])**2)
            # Huber Loss
            delta = 20.0
            if dist < delta:
                loss = 0.5 * (dist**2)
            else:
                loss = delta * (dist - 0.5 * delta)
            total_loss += loss
        return total_loss

    guess = [N_f, E_f, alt_g, phi]
    bounds = [
        (-50.0, 50.0),
        (-100.0, 0.0),
        (130.0, 180.0),
        (np.radians(45.0), np.radians(75.0))
    ]
    
    res = opt.minimize(loss_func, guess, args=(inliers,), bounds=bounds, method='L-BFGS-B')
    if res.success:
        n_f, e_f, a_g, p = res.x
        d_f = alt0 - a_g
        flag_lat, flag_lon, flag_alt = pm.ned2geodetic(n_f, e_f, d_f, lat0, lon0, alt0)
        rmse = np.sqrt(res.fun / len(inliers))
        
        print(f"\n=== REFINED ROBUST FIT ON INLIERS ===")
        print(f"Optimal Local Parameters:")
        print(f"  Flag NED North:  {n_f:.4f} m")
        print(f"  Flag NED East:   {e_f:.4f} m")
        print(f"  Ground Altitude: {a_g:.4f} m (MSL)")
        print(f"  Camera Yaw Offset (phi): {np.degrees(p):.2f} deg")
        print(f"\nFlag Geographic Location:")
        print(f"  Latitude:  {flag_lat:.8f} N")
        print(f"  Longitude: {flag_lon:.8f} E")
        print(f"  Altitude:  {flag_alt:.2f} m (MSL)")
        print(f"  RMSE:      {rmse:.2f} px")
        
        # Save results
        with open('estimated_flag_location.txt', 'w') as out_f:
            out_f.write("=== FINAL FLAG LOCATION ===\n")
            out_f.write(f"Flag Latitude:  {flag_lat:.8f} N\n")
            out_f.write(f"Flag Longitude: {flag_lon:.8f} E\n")
            out_f.write(f"Flag Altitude:  {flag_alt:.2f} m\n\n")
            out_f.write("=== CALIBRATION PARAMETERS ===\n")
            out_f.write(f"Flag NED North:  {n_f:.4f} m\n")
            out_f.write(f"Flag NED East:   {e_f:.4f} m\n")
            out_f.write(f"Ground Altitude: {a_g:.4f} m\n")
            out_f.write(f"Camera Yaw Offset (phi): {np.degrees(p):.2f} deg\n")
            out_f.write(f"Reprojection RMSE: {rmse:.2f} px\n")
            out_f.write(f"Inliers: {len(inliers)} / {len(data_points)}\n")
        print("\nResults written to estimated_flag_location.txt")
        
if __name__ == '__main__':
    main()
