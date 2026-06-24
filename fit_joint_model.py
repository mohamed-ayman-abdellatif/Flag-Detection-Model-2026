import json
import os
import numpy as np
import scipy.optimize as opt
import pymap3d as pm

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

def joint_loss_function(params, data_dyn, data_stat, f_dyn, f_stat, W_dyn, H_dyn, W_stat, H_stat):
    # params: [N_f, E_f, alt_g, phi_dyn, theta_stat]
    N_f, E_f, alt_g, phi_dyn, theta_stat = params
    
    total_loss = 0
    
    # 1. Dynamic points loss
    x0_dyn = W_dyn / 2.0
    y0_dyn = H_dyn / 2.0
    for dp in data_dyn:
        N_d = dp['n_d']
        E_d = dp['e_d']
        alt_d = dp['alt_d']
        yaw_d = dp['yaw_d']
        x_meas = dp['x_pixel']
        y_meas = dp['y_pixel']
        
        theta = yaw_d + phi_dyn
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        h = alt_d - alt_g
        if h <= 0.1:
            return 1e12
            
        dy = N_f - N_d
        dx = E_f - E_d
        
        X_c = dx * cos_t - dy * sin_t
        Y_c = -dx * sin_t - dy * cos_t
        
        x_proj = x0_dyn + f_dyn * X_c / h
        y_proj = y0_dyn + f_dyn * Y_c / h
        
        dist = np.sqrt((x_proj - x_meas)**2 + (y_proj - y_meas)**2)
        
        # Huber Loss
        delta = 20.0
        if dist < delta:
            loss = 0.5 * (dist**2)
        else:
            loss = delta * (dist - 0.5 * delta)
        total_loss += loss
        
    # 2. Stationary points loss
    x0_stat = W_stat / 2.0
    y0_stat = H_stat / 2.0
    cos_ts = np.cos(theta_stat)
    sin_ts = np.sin(theta_stat)
    
    for dp in data_stat:
        N_d = dp['n_d']
        E_d = dp['e_d']
        alt_d = dp['alt_d']
        x_meas = dp['x_pixel']
        y_meas = dp['y_pixel']
        
        h = alt_d - alt_g
        if h <= 0.1:
            return 1e12
            
        dy = N_f - N_d
        dx = E_f - E_d
        
        X_c = dx * cos_ts - dy * sin_ts
        Y_c = -dx * sin_ts - dy * cos_ts
        
        x_proj = x0_stat + f_stat * X_c / h
        y_proj = y0_stat + f_stat * Y_c / h
        
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
    gps_json_dyn = r'meta_data\GX014208_1_GPS9.json'
    gps_json_stat = r'meta_data\GX014209_1_GPS9.json'
    
    detections = load_detections()
    dets_dyn = detections['GL014208.LRV']
    dets_stat = detections['GX014209.MP4']
    
    # We only use moving sequence for dyn (t between 100.0s and 140.0s)
    dets_dyn_filtered = [d for d in dets_dyn if 100.0 <= d['t_sec'] <= 140.0]
    
    gps_dyn = sorted(load_gps(gps_json_dyn), key=lambda x: x['cts'])
    gps_stat = sorted(load_gps(gps_json_stat), key=lambda x: x['cts'])
    
    # We define a global origin using the first GPS of dyn
    lat0 = gps_dyn[0]['lat']
    lon0 = gps_dyn[0]['lon']
    alt0 = gps_dyn[0]['alt']
    
    # Dynamic telemetry preparation
    gps_n_d, gps_e_d, gps_d_d, gps_t_d = [], [], [], []
    for g in gps_dyn:
        n, e, d = pm.geodetic2ned(g['lat'], g['lon'], g['alt'], lat0, lon0, alt0)
        gps_n_d.append(n)
        gps_e_d.append(e)
        gps_d_d.append(d)
        gps_t_d.append(g['cts'] / 1000.0)
    gps_n_d = np.array(gps_n_d)
    gps_e_d = np.array(gps_e_d)
    gps_d_d = np.array(gps_d_d)
    gps_t_d = np.array(gps_t_d)
    
    headings_dyn = []
    n_samples_dyn = len(gps_dyn)
    for idx in range(n_samples_dyn):
        start_idx = max(0, idx - 10)
        end_idx = min(n_samples_dyn - 1, idx + 10)
        dn = gps_n_d[end_idx] - gps_n_d[start_idx]
        de = gps_e_d[end_idx] - gps_e_d[start_idx]
        if np.sqrt(dn**2 + de**2) > 0.1:
            headings_dyn.append(np.atan2(de, dn))
        else:
            headings_dyn.append(0.0)
    headings_dyn = np.array(headings_dyn)
    
    last_h = 0.0
    for idx in range(n_samples_dyn):
        if headings_dyn[idx] == 0.0:
            headings_dyn[idx] = last_h
        else:
            last_h = headings_dyn[idx]
            
    # Interpolate dyn points
    data_dyn = []
    for d in dets_dyn_filtered:
        t_sec = d['t_sec']
        n_interp = np.interp(t_sec, gps_t_d, gps_n_d)
        e_interp = np.interp(t_sec, gps_t_d, gps_e_d)
        d_interp = np.interp(t_sec, gps_t_d, gps_d_d)
        alt_interp = alt0 - d_interp
        yaw_interp = np.interp(t_sec, gps_t_d, headings_dyn)
        
        data_dyn.append({
            'n_d': n_interp,
            'e_d': e_interp,
            'alt_d': alt_interp,
            'yaw_d': yaw_interp,
            'x_pixel': d['x'],
            'y_pixel': d['y']
        })
        
    # Stationary telemetry preparation
    gps_n_s, gps_e_s, gps_d_s, gps_t_s = [], [], [], []
    for g in gps_stat:
        n, e, d = pm.geodetic2ned(g['lat'], g['lon'], g['alt'], lat0, lon0, alt0)
        gps_n_s.append(n)
        gps_e_s.append(e)
        gps_d_s.append(d)
        gps_t_s.append(g['cts'] / 1000.0)
    gps_n_s = np.array(gps_n_s)
    gps_e_s = np.array(gps_e_s)
    gps_d_s = np.array(gps_d_s)
    gps_t_s = np.array(gps_t_s)
    
    # Interpolate stat points
    data_stat = []
    for d in dets_stat:
        t_sec = d['t_sec']
        n_interp = np.interp(t_sec, gps_t_s, gps_n_s)
        e_interp = np.interp(t_sec, gps_t_s, gps_e_s)
        d_interp = np.interp(t_sec, gps_t_s, gps_d_s)
        alt_interp = alt0 - d_interp
        
        data_stat.append({
            'n_d': n_interp,
            'e_d': e_interp,
            'alt_d': alt_interp,
            'x_pixel': d['x'],
            'y_pixel': d['y']
        })
        
    print(f"Dynamic points: {len(data_dyn)}, Stationary points: {len(data_stat)}")
    
    # Resolutions & Focal Lengths
    W_dyn, H_dyn = 768.0, 432.0
    f_dyn = 3000.0 * (W_dyn / 3840.0) # 600.0
    
    W_stat, H_stat = 3840.0, 2160.0
    f_stat = 3000.0
    
    # Multi-start global optimization
    best_loss = 1e15
    best_res = None
    
    np.random.seed(42)
    for i in range(100):
        # random guesses
        # N_f around -10m, E_f around -50m (from moving sequence)
        init_N = np.random.uniform(-40.0, 20.0)
        init_E = np.random.uniform(-100.0, 0.0)
        init_alt_g = np.random.uniform(alt0 - 150.0, alt0 - 20.0)
        init_phi = np.random.uniform(-np.pi, np.pi)
        init_theta = np.random.uniform(-np.pi, np.pi)
        
        guess = [init_N, init_E, init_alt_g, init_phi, init_theta]
        bounds = [
            (-200.0, 200.0),
            (-200.0, 200.0),
            (alt0 - 180.0, alt0 - 10.0),
            (-2.0*np.pi, 2.0*np.pi),
            (-2.0*np.pi, 2.0*np.pi)
        ]
        
        res = opt.minimize(
            joint_loss_function, guess, 
            args=(data_dyn, data_stat, f_dyn, f_stat, W_dyn, H_dyn, W_stat, H_stat),
            bounds=bounds, method='L-BFGS-B'
        )
        
        if res.success and res.fun < best_loss:
            best_loss = res.fun
            best_res = res
            
    if best_res is not None:
        N_f, E_f, alt_g, phi_dyn, theta_stat = best_res.x
        d_f = alt0 - alt_g
        flag_lat, flag_lon, flag_alt = pm.ned2geodetic(N_f, E_f, d_f, lat0, lon0, alt0)
        rmse_dyn = np.sqrt(loss_function_single(N_f, E_f, alt_g, phi_dyn, data_dyn, f_dyn, W_dyn, H_dyn) / len(data_dyn))
        rmse_stat = np.sqrt(loss_function_single_stat(N_f, E_f, alt_g, theta_stat, data_stat, f_stat, W_stat, H_stat) / len(data_stat))
        
        print("\n=== JOINT OPTIMIZATION SUCCEEDED ===")
        print(f"Optimal Local Parameters:")
        print(f"  Flag NED North:  {N_f:.4f} m")
        print(f"  Flag NED East:   {E_f:.4f} m")
        print(f"  Ground Altitude: {alt_g:.4f} m (MSL)")
        print(f"  Dynamic Phi:     {np.degrees(phi_dyn):.2f} deg")
        print(f"  Stationary Yaw:  {np.degrees(theta_stat):.2f} deg")
        
        print(f"\nFlag Geographic Location:")
        print(f"  Latitude:  {flag_lat:.8f} N")
        print(f"  Longitude: {flag_lon:.8f} E")
        print(f"  Altitude:  {flag_alt:.2f} m (MSL)")
        
        print(f"\nReprojection RMSE:")
        print(f"  Dynamic Pass:    {rmse_dyn:.2f} px")
        print(f"  Stationary Pass: {rmse_stat:.2f} px")
        
        # Save results
        with open('estimated_flag_location.txt', 'w') as out_f:
            out_f.write("=== FINAL FLAG LOCATION ===\n")
            out_f.write(f"Flag Latitude:  {flag_lat:.8f} N\n")
            out_f.write(f"Flag Longitude: {flag_lon:.8f} E\n")
            out_f.write(f"Flag Altitude:  {flag_alt:.2f} m\n\n")
            out_f.write("=== CALIBRATION PARAMETERS ===\n")
            out_f.write(f"Flag NED North:  {N_f:.4f} m\n")
            out_f.write(f"Flag NED East:   {E_f:.4f} m\n")
            out_f.write(f"Ground Altitude: {alt_g:.4f} m\n")
            out_f.write(f"Dynamic Phi:     {np.degrees(phi_dyn):.2f} deg\n")
            out_f.write(f"Stationary Yaw:  {np.degrees(theta_stat):.2f} deg\n")
            out_f.write(f"Dynamic RMSE:    {rmse_dyn:.2f} px\n")
            out_f.write(f"Stationary RMSE: {rmse_stat:.2f} px\n")
        print("\nResults written to estimated_flag_location.txt")

def loss_function_single(N_f, E_f, alt_g, phi, data_points, f, W, H):
    x_0, y_0 = W/2.0, H/2.0
    total_loss = 0
    for dp in data_points:
        theta = dp['yaw_d'] + phi
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        h = dp['alt_d'] - alt_g
        X_c = (E_f - dp['e_d']) * cos_t - (N_f - dp['n_d']) * sin_t
        Y_c = -(E_f - dp['e_d']) * sin_t - (N_f - dp['n_d']) * cos_t
        x_proj = x_0 + f * X_c / h
        y_proj = y_0 + f * Y_c / h
        total_loss += (x_proj - dp['x_pixel'])**2 + (y_proj - dp['y_pixel'])**2
    return total_loss

def loss_function_single_stat(N_f, E_f, alt_g, theta, data_points, f, W, H):
    x_0, y_0 = W/2.0, H/2.0
    total_loss = 0
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    for dp in data_points:
        h = dp['alt_d'] - alt_g
        X_c = (E_f - dp['e_d']) * cos_t - (N_f - dp['n_d']) * sin_t
        Y_c = -(E_f - dp['e_d']) * sin_t - (N_f - dp['n_d']) * cos_t
        x_proj = x_0 + f * X_c / h
        y_proj = y_0 + f * Y_c / h
        total_loss += (x_proj - dp['x_pixel'])**2 + (y_proj - dp['y_pixel'])**2
    return total_loss

if __name__ == '__main__':
    main()
