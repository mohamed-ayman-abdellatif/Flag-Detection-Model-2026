import json
import os
import numpy as np
import scipy.optimize as opt
import pymap3d as pm

# 1. Flag detections in GL014208.LRV during the moving sequence (t=100.1s to 132.1s)
# Hardcoded to ensure robustness against file-system/log changes
DETECTIONS = [
    {'frame': 3000, 't_sec': 100.1, 'x': 77.4, 'y': 387.0, 'area': 7.0},
    {'frame': 3030, 't_sec': 101.1, 'x': 62.2, 'y': 203.2, 'area': 12.0},
    {'frame': 3090, 't_sec': 103.1, 'x': 55.9, 'y': 219.0, 'area': 14.5},
    {'frame': 3330, 't_sec': 111.1, 'x': 132.7, 'y': 149.2, 'area': 15.0},
    {'frame': 3480, 't_sec': 116.1, 'x': 64.0, 'y': 227.9, 'area': 5.5},
    {'frame': 3540, 't_sec': 118.1, 'x': 65.7, 'y': 245.3, 'area': 7.5},
    {'frame': 3750, 't_sec': 125.1, 'x': 671.0, 'y': 333.8, 'area': 4.0},
    {'frame': 3840, 't_sec': 128.1, 'x': 86.3, 'y': 86.0, 'area': 7.5},
    {'frame': 3900, 't_sec': 130.1, 'x': 111.0, 'y': 244.8, 'area': 8.5},
    {'frame': 3960, 't_sec': 132.1, 'x': 113.1, 'y': 255.0, 'area': 14.0}
]

def load_gps_json(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"GPS JSON file not found at: {json_path}")
        
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # GoPro JSON structure: {"1": {"streams": {"GPS9": {"samples": [...]}}}}
    streams = data.get("1", {}).get("streams", {})
    if "GPS9" not in streams:
        for k in data.keys():
            streams = data[k].get("streams", {})
            if "GPS9" in streams:
                break
                
    if "GPS9" not in streams:
        raise ValueError("GPS9 stream not found in JSON metadata")
        
    samples = streams["GPS9"]["samples"]
    gps_data = []
    for s in samples:
        val = s["value"] # [lat, lon, alt, spd2d, spd3d, days, secs, dop, fix]
        gps_data.append({
            'cts': s['cts'], # ms
            'lat': val[0],
            'lon': val[1],
            'alt': val[2],
            'fix': val[8]
        })
    return sorted(gps_data, key=lambda x: x['cts'])

def loss_function(params, data_points, W, H, f, phi, alt_g):
    N_f, E_f = params
    x_0 = W / 2.0
    y_0 = H / 2.0
    
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
        dy = N_f - N_d
        dx = E_f - E_d
        
        # Project onto camera coordinates
        X_c = dx * cos_t - dy * sin_t
        Y_c = -dx * sin_t - dy * cos_t
        
        x_proj = x_0 + f * X_c / h
        y_proj = y_0 + f * Y_c / h
        
        dist = np.sqrt((x_proj - x_meas)**2 + (y_proj - y_meas)**2)
        total_loss += dist**2
        
    return total_loss

def main():
    print("=== Geometric Flag Localization (New Flight Dataset) ===")
    
    # Path to pre-extracted telemetry
    gps_json_path = r'C:\Users\mido\Documents\antigravity\focused-babbage\meta_data\GX014208_1_GPS9.json'
    print(f"Loading GPS9 telemetry from: {gps_json_path}")
    gps_data = load_gps_json(gps_json_path)
    
    lat0, lon0, alt0 = gps_data[0]['lat'], gps_data[0]['lon'], gps_data[0]['alt']
    print(f"Takeoff (Origin) coordinates: Lat={lat0:.8f}, Lon={lon0:.8f}, Alt={alt0:.2f}m")
    
    # Convert GPS data to local NED coordinates
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
    
    # Estimate heading from drone velocity vectors
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
            
    # Camera intrinsic properties for GL014208.LRV
    # LRV Resolution: 768 x 432
    W = 768.0
    H = 432.0
    f = 600.0 # corresponds to f_4k = 3000 px scaled by (768 / 3840)
    
    # Match detections with telemetry
    data_points = []
    for d in DETECTIONS:
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
            'y_pixel': d['y']
        })
        
    print(f"Loaded {len(data_points)} matched flag detections in moving sequence.")
    
    # Set physical ground plane and mounting constraints:
    # 1. Ground altitude = 224.0 m MSL (Takeoff altitude ~ 230 m)
    # 2. Camera yaw offset phi = 90.0 degrees (sideways-pointing camera)
    alt_g = 224.0
    phi = np.radians(90.0)
    
    # Run localization optimization (L-BFGS-B)
    guess = [0.0, 0.0]
    bounds = [
        (-200.0, 200.0),
        (-200.0, 200.0)
    ]
    res = opt.minimize(loss_function, guess, args=(data_points, W, H, f, phi, alt_g), bounds=bounds, method='L-BFGS-B')
    
    if res.success:
        N_f, E_f = res.x
        d_f = alt0 - alt_g
        flag_lat, flag_lon, flag_alt = pm.ned2geodetic(N_f, E_f, d_f, lat0, lon0, alt0)
        rmse = np.sqrt(res.fun / len(data_points))
        
        print("\n=== CONSTRAINED FIT SUCCESSFUL ===")
        print(f"Optimal Parameters:")
        print(f"  Flag NED North:  {N_f:.4f} m")
        print(f"  Flag NED East:   {E_f:.4f} m")
        print(f"  Ground Altitude: {alt_g:.4f} m (MSL)")
        print(f"  Camera Yaw Offset (phi): 90.00 deg (fixed)")
        print(f"  Reprojection RMSE: {rmse:.2f} px")
        
        print(f"\nFinal Estimated Flag Coordinates:")
        print(f"  Latitude:  {flag_lat:.8f} N")
        print(f"  Longitude: {flag_lon:.8f} E")
        print(f"  Altitude:  {flag_alt:.2f} m (MSL)")
        
        # Write outputs to file
        output_path = 'estimated_flag_location.txt'
        with open(output_path, 'w') as out_f:
            out_f.write("=== FINAL FLAG LOCATION ===\n")
            out_f.write(f"Flag Latitude:  {flag_lat:.8f} N\n")
            out_f.write(f"Flag Longitude: {flag_lon:.8f} E\n")
            out_f.write(f"Flag Altitude:  {flag_alt:.2f} m\n\n")
            out_f.write("=== CALIBRATION PARAMETERS ===\n")
            out_f.write(f"Flag NED North:  {N_f:.4f} m\n")
            out_f.write(f"Flag NED East:   {E_f:.4f} m\n")
            out_f.write(f"Ground Altitude: {alt_g:.4f} m\n")
            out_f.write(f"Camera Yaw Offset (phi): 90.00 deg\n")
            out_f.write(f"Reprojection RMSE: {rmse:.2f} px\n")
            out_f.write(f"Inliers: {len(data_points)} / {len(data_points)} (Moving sequence frames 3000-3960)\n")
            
        print(f"\nResults successfully written to {output_path}")
    else:
        print("\nOptimization failed!")

if __name__ == '__main__':
    main()
