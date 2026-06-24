import csv
import json
import os
import numpy as np
import scipy.optimize as opt
import pymap3d as pm
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
                'q': [float(row['qx']), float(row['qy']), float(row['qz']), float(row['qw'])] # scipy is [x,y,z,w]
            })
    return data

def loss_function(params, data_points, W, H):
    # params: [N_f, E_f, alt_g, f, yaw_startup, x_0, y_0]
    N_f, E_f, alt_g, f, yaw_startup, x_0, y_0 = params
    
    total_loss = 0
    
    # R_z(yaw_startup)
    cos_ys = np.cos(yaw_startup)
    sin_ys = np.sin(yaw_startup)
    R_z = np.array([
        [cos_ys, -sin_ys, 0],
        [sin_ys, cos_ys, 0],
        [0, 0, 1]
    ])
    
    # R0 maps Camera to NED when yaw_startup=0
    # X_cam = East, Y_cam = South, Z_cam = Down
    # So Camera to NED is: North = -Y_cam, East = X_cam, Down = Z_cam
    R0 = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    
    R_world_to_initial = R_z @ R0
    
    for dp in data_points:
        N_d = dp['n_d']
        E_d = dp['e_d']
        alt_d = dp['alt_d']
        x_meas = dp['x_pixel']
        y_meas = dp['y_pixel']
        
        q_rel = dp['q_rel']
        # Convert q_rel to rotation matrix (maps current cam to initial cam)
        R_rel = R.from_quat(q_rel).as_matrix()
        
        # Camera to NED rotation matrix
        R_cam_to_ned = R_world_to_initial @ R_rel
        
        # NED to Camera rotation matrix
        R_ned_to_cam = R_cam_to_ned.T
        
        # Displacement vector in NED
        v_NED = np.array([N_f - N_d, E_f - E_d, alt_d - alt_g]) # Down = -(alt_g - alt_d) = alt_d - alt_g
        
        # Transform to camera frame
        v_cam = R_ned_to_cam @ v_NED
        
        x_c, y_c, z_c = v_cam
        if z_c <= 0.1:
            return 1e12
            
        # Pinhole projection
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

def fit_with_cori_data(video_name, gps_json_path, cori_csv_path):
    print(f"\n========================================")
    print(f"Fitting {video_name} with CORI...")
    print(f"========================================")
    
    detections = load_detections()[video_name]
    gps_data = sorted(load_gps(gps_json_path), key=lambda x: x['cts'])
    cori_data = sorted(load_cori(cori_csv_path), key=lambda x: x['cts_ms'])
    
    lat0, lon0, alt0 = gps_data[0]['lat'], gps_data[0]['lon'], gps_data[0]['alt']
    
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
    
    import cv2
    cap = cv2.VideoCapture(r'C:\Users\mido\Documents\antigravity\focused-babbage\test_flight\\' + video_name)
    W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    
    f_init = 3000.0 * (W / 3840.0)
    
    # Interpolate state for each detection
    data_points = []
    for d in detections:
        t_sec = d['t_sec']
        
        n_interp = np.interp(t_sec, gps_t, gps_n)
        e_interp = np.interp(t_sec, gps_t, gps_e)
        d_interp = np.interp(t_sec, gps_t, gps_d)
        alt_interp = alt0 - d_interp
        
        # Quaternion interpolation: we can just find nearest index since CORI is 30Hz
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
            'y_pixel': d['y'],
            'area': d['area']
        })
        
    print(f"Loaded {len(data_points)} data points.")
    
    # Grid search for ground alt and yaw_startup
    best_loss = 1e15
    best_init = None
    
    ground_alts = np.linspace(alt0 - 150, alt0 - 20, 27)
    yaw_startups = np.linspace(-np.pi, np.pi, 37)
    
    # Let's search over a small grid of (N_f, E_f) based on the drone's position during detections
    # In GL014208, the drone is near (-30, -11)
    # Let's search over (N_f, E_f) around (-50, 50)
    
    for alt_g_val in ground_alts:
        for ys in yaw_startups:
            # We can solve for (N_f, E_f) using linear least squares!
            # Let's formulate the projection equation:
            # R_ned_to_cam = R_rel.T @ R0.T @ R_z.T
            # v_cam = R_ned_to_cam @ [N_f - N_d, E_f - E_d, alt_d - alt_g]
            # Since R_ned_to_cam is a 3x3 matrix:
            # Let R = R_ned_to_cam
            # v_cam = R @ [N_f - N_d, E_f - E_d, alt_d - alt_g].T
            # Let's write the projection equation:
            # x_proj - x_0 = f * (R_00*(N_f-N_d) + R_01*(E_f-E_d) + R_02*(alt_d-alt_g)) / (R_20*(N_f-N_d) + R_21*(E_f-E_d) + R_22*(alt_d-alt_g))
            # This is non-linear. But we can solve it approximately or just evaluate loss.
            # Let's evaluate loss for a grid of (N_f, E_f) for initialization!
            # Since evaluating the loss function is very fast (only 43 points), we can just do a grid search over N_f, E_f as well!
            # Range: N_f in [-60, 20], E_f in [-40, 40]
            pass
            
    # Actually, let's write a simple random search or optimization from multiple initial guesses!
    # Let's sample initial guesses:
    print("Running multi-start optimization...")
    best_res = None
    best_fun = 1e15
    
    # Generate random initial guesses
    np.random.seed(42)
    for _ in range(50):
        init_N = np.random.uniform(-100.0, 100.0)
        init_E = np.random.uniform(-100.0, 100.0)
        init_alt_g = np.random.uniform(alt0 - 150.0, alt0 - 20.0)
        init_f = np.random.uniform(0.8 * f_init, 1.2 * f_init)
        init_ys = np.random.uniform(-np.pi, np.pi)
        
        guess = [init_N, init_E, init_alt_g, init_f, init_ys, W/2.0, H/2.0]
        
        bounds = [
            (-500.0, 500.0),
            (-500.0, 500.0),
            (alt0 - 180.0, alt0 - 10.0),
            (0.5 * f_init, 2.0 * f_init),
            (-2.0 * np.pi, 2.0 * np.pi),
            (W/2.0 - 50.0, W/2.0 + 50.0),
            (H/2.0 - 50.0, H/2.0 + 50.0)
        ]
        
        res = opt.minimize(loss_function, guess, args=(data_points, W, H), bounds=bounds, method='L-BFGS-B')
        if res.success and res.fun < best_fun:
            best_fun = res.fun
            best_res = res
            
    if best_res is None:
        print("Optimization failed.")
        return
        
    N_f, E_f, alt_g, f_val, yaw_startup, x_0, y_0 = best_res.x
    d_f = alt0 - alt_g
    flag_lat, flag_lon, flag_alt = pm.ned2geodetic(N_f, E_f, d_f, lat0, lon0, alt0)
    rmse = np.sqrt(best_res.fun / len(data_points))
    
    print(f"\n=== CORI Optimization Succeeded ===")
    print(f"Optimal Local Parameters:")
    print(f"  Flag NED North:  {N_f:.4f} m")
    print(f"  Flag NED East:   {E_f:.4f} m")
    print(f"  Ground Altitude: {alt_g:.4f} m (MSL)")
    print(f"  Focal Length:    {f_val:.2f} px")
    print(f"  Startup Yaw:     {np.degrees(yaw_startup):.2f} deg")
    print(f"  Principal Point: ({x_0:.2f}, {y_0:.2f})")
    print(f"\nFlag Geographic Location:")
    print(f"  Latitude:  {flag_lat:.8f} N")
    print(f"  Longitude: {flag_lon:.8f} E")
    print(f"  Altitude:  {flag_alt:.2f} m (MSL)")
    print(f"  RMSE:      {rmse:.2f} px")
    
    # Print errors and filter outliers
    print("\nReprojection Errors:")
    inliers = []
    for dp in data_points:
        cos_ys = np.cos(yaw_startup)
        sin_ys = np.sin(yaw_startup)
        R_z = np.array([
            [cos_ys, -sin_ys, 0],
            [sin_ys, cos_ys, 0],
            [0, 0, 1]
        ])
        R0 = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        R_world_to_initial = R_z @ R0
        R_rel = R.from_quat(dp['q_rel']).as_matrix()
        R_cam_to_ned = R_world_to_initial @ R_rel
        R_ned_to_cam = R_cam_to_ned.T
        
        v_NED = np.array([N_f - dp['n_d'], E_f - dp['e_d'], dp['alt_d'] - alt_g])
        v_cam = R_ned_to_cam @ v_NED
        x_proj = x_0 + f_val * v_cam[0] / v_cam[2]
        y_proj = y_0 + f_val * v_cam[1] / v_cam[2]
        
        dist = np.sqrt((x_proj - dp['x_pixel'])**2 + (y_proj - dp['y_pixel'])**2)
        print(f"  Frame {dp['frame']:5d} ({dp['t_sec']:5.1f}s): Error = {dist:6.2f} px, Area = {dp['area']:.1f}")
        
        # Threshold at 30 pixels (less than 4% of image size)
        if dist < 30.0:
            inliers.append(dp)
            
    print(f"\nInliers count: {len(inliers)} / {len(data_points)}")
    
    if len(inliers) < 3:
        print("Not enough inliers to re-fit.")
        return
        
    # Re-fit on inliers
    print("\nRe-fitting on inliers...")
    best_res_inliers = None
    best_fun_inliers = 1e15
    for _ in range(50):
        init_N = np.random.uniform(N_f - 20, N_f + 20)
        init_E = np.random.uniform(E_f - 20, E_f + 20)
        init_alt_g = np.random.uniform(alt_g - 10, alt_g + 10)
        init_f = np.random.uniform(f_val - 200, f_val + 200)
        init_ys = np.random.uniform(yaw_startup - 0.2, yaw_startup + 0.2)
        
        guess = [init_N, init_E, init_alt_g, init_f, init_ys, x_0, y_0]
        
        bounds = [
            (N_f - 50.0, N_f + 50.0),
            (E_f - 50.0, E_f + 50.0),
            (alt_g - 20.0, alt_g + 20.0),
            (0.5 * f_init, 2.0 * f_init),
            (-2.0 * np.pi, 2.0 * np.pi),
            (x_0 - 20.0, x_0 + 20.0),
            (y_0 - 20.0, y_0 + 20.0)
        ]
        
        res = opt.minimize(loss_function, guess, args=(inliers, W, H), bounds=bounds, method='L-BFGS-B')
        if res.success and res.fun < best_fun_inliers:
            best_fun_inliers = res.fun
            best_res_inliers = res
            
    if best_res_inliers is not None:
        N_f, E_f, alt_g, f_val, yaw_startup, x_0, y_0 = best_res_inliers.x
        d_f = alt0 - alt_g
        flag_lat, flag_lon, flag_alt = pm.ned2geodetic(N_f, E_f, d_f, lat0, lon0, alt0)
        rmse = np.sqrt(best_res_inliers.fun / len(inliers))
        
        print(f"\n=== FINAL REFINED FIT ON INLIERS ===")
        print(f"Optimal Local Parameters:")
        print(f"  Flag NED North:  {N_f:.4f} m")
        print(f"  Flag NED East:   {E_f:.4f} m")
        print(f"  Ground Altitude: {alt_g:.4f} m (MSL)")
        print(f"  Focal Length:    {f_val:.2f} px")
        print(f"  Startup Yaw:     {np.degrees(yaw_startup):.2f} deg")
        print(f"  Principal Point: ({x_0:.2f}, {y_0:.2f})")
        print(f"\nFlag Geographic Location:")
        print(f"  Latitude:  {flag_lat:.8f} N")
        print(f"  Longitude: {flag_lon:.8f} E")
        print(f"  Altitude:  {flag_alt:.2f} m (MSL)")
        print(f"  RMSE:      {rmse:.2f} px")

if __name__ == '__main__':
    fit_with_cori_data('GL014208.LRV', r'meta_data\GX014208_1_GPS9.json', 'GL014208_CORI.csv')
