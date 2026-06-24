import json
import os
import numpy as np
import scipy.optimize as opt
import pymap3d as pm

def load_detections():
    log_path = r'C:\Users\mido\.gemini\antigravity\brain\7c583763-3713-4256-b5ad-dfa91099fc03\.system_generated\tasks\task-287.log'
    if not os.path.exists(log_path):
        raise FileNotFoundError("Log file not found!")

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

def loss_function(params, data_points):
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

def fit_for_video(video_name, gps_json_path, image_width=1920.0, image_height=1080.0):
    print(f"\n========================================")
    print(f"Fitting for {video_name}...")
    print(f"========================================")
    
    detections = load_detections()
    if video_name not in detections:
        print(f"No detections found for {video_name}")
        return
        
    dets = detections[video_name]
    gps_data = load_gps(gps_json_path)
    
    # Sort GPS by cts
    gps_data = sorted(gps_data, key=lambda x: x['cts'])
    
    # Origin is first GPS point
    lat0 = gps_data[0]['lat']
    lon0 = gps_data[0]['lon']
    alt0 = gps_data[0]['alt']
    
    # Calculate NED coordinates for all GPS samples
    gps_n = []
    gps_e = []
    gps_d = []
    gps_t = []
    for g in gps_data:
        n, e, d = pm.geodetic2ned(g['lat'], g['lon'], g['alt'], lat0, lon0, alt0)
        gps_n.append(n)
        gps_e.append(e)
        gps_d.append(d)
        gps_t.append(g['cts'] / 1000.0) # seconds
        
    gps_n = np.array(gps_n)
    gps_e = np.array(gps_e)
    gps_d = np.array(gps_d)
    gps_t = np.array(gps_t)
    
    # Compute headings for all GPS samples
    # We can estimate heading from velocity vector
    headings = []
    n_samples = len(gps_data)
    for idx in range(n_samples):
        start_idx = max(0, idx - 10) # smooth a bit more since GPS is 10Hz
        end_idx = min(n_samples - 1, idx + 10)
        dn = gps_n[end_idx] - gps_n[start_idx]
        de = gps_e[end_idx] - gps_e[start_idx]
        if np.sqrt(dn**2 + de**2) > 0.1:
            headings.append(np.atan2(de, dn))
        else:
            headings.append(0.0)
            
    # Fill in static values
    headings = np.array(headings)
    last_h = 0.0
    for idx in range(n_samples):
        if headings[idx] == 0.0:
            headings[idx] = last_h
        else:
            last_h = headings[idx]
            
    # Interpolate drone state to detection timestamps
    data_points = []
    for d in dets:
        t_sec = d['t_sec']
        
        # Interpolate position
        n_interp = np.interp(t_sec, gps_t, gps_n)
        e_interp = np.interp(t_sec, gps_t, gps_e)
        d_interp = np.interp(t_sec, gps_t, gps_d)
        alt_interp = alt0 - d_interp
        
        # Heading interpolation (taking care of wrapping is ideal, but let's simple linear for now)
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
        
    print(f"Number of data points: {len(data_points)}")
    
    # Grid search for initialization
    best_loss = 1e15
    best_init = None
    
    # Search range for ground alt (takeoff alt0 - ground_alt should be around 77m like first flight, so ground alt around alt0 - 77)
    # Let's search over ground alt from alt0 - 150 to alt0 - 30
    ground_alts = np.linspace(alt0 - 120, alt0 - 20, 21)
    # Search over yaw offset phi
    phis = np.linspace(-np.pi, np.pi, 37)
    
    # Setup focal length search or fixed focal length?
    # In first flight, f = 3000. Let's start with 3000. Wait, for 4K video, f is twice as big as 1080p.
    # Wait, LRV video is 1280x720. If we use LRV detections, the image width is 1280, height is 720.
    # The focal length scales with image width!
    # If standard 4K (3840 width) focal length is 3000, then for LRV (1280 width), the focal length is:
    # 3000 * (1280 / 3840) = 1000 pixels!
    # Let's check what image size the detections were found in.
    # In scan_video_for_flag.py, the input image was parsed at native resolution.
    # Wait! Sequential Scanning printed:
    # `GL014208.LRV: Center=(25.0, 246.2)` -> values are within [0..1280] or [0..1920]?
    # Let's check: `Center=(25.0, 246.2)` in GL014208.LRV, and in GX014209.MP4 it's `Center=(297.1, 1346.2)`.
    # Wait! In GL014208.LRV, the center coordinates are smaller. So the image width for GL014208.LRV is indeed smaller (likely 1920 or 1280 or 1080).
    # Let's write code to get the resolution of the video.
    # Let's see: we can search over focal length f, or scale it.
    # Let's assume:
    # - For GL014208.LRV, let's use the width and height of the video.
    # - Let's search over f.
    
    import cv2
    cap = cv2.VideoCapture(r'C:\Users\mido\Documents\antigravity\focused-babbage\test_flight\\' + video_name)
    W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    print(f"Video resolution: {W} x {H}")
    
    # Scale focal length accordingly:
    # Standard: 3000 pixels for 3840 width.
    f_init = 3000.0 * (W / 3840.0)
    print(f"Focal length initial guess: {f_init}")
    
    # Loop for least squares grid search
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
            loss = loss_function([xy_f[1], xy_f[0], alt_g_val, f_init, phi_val, W/2.0, H/2.0], data_points)
            if loss < best_loss:
                best_loss = loss
                best_init = [xy_f[1], xy_f[0], alt_g_val, f_init, phi_val, W/2.0, H/2.0]
                
    if best_init is None:
        print("Initialization failed.")
        return
        
    print(f"Initialization best loss: {best_loss:.2f}")
    print(f"Init parameters: N={best_init[0]:.2f}, E={best_init[1]:.2f}, AltG={best_init[2]:.2f}, f={best_init[3]:.2f}, phi={np.degrees(best_init[4]):.2f} deg")
    
    # Run optimizer
    # Bounds:
    # N_f: [-500, 500] relative to start
    # E_f: [-500, 500] relative to start
    # alt_g: [alt0 - 150, alt0 - 10]
    # f: [0.5 * f_init, 2.0 * f_init]
    # phi: [-2*pi, 2*pi]
    # x_0: [W/2 - 100, W/2 + 100]
    # y_0: [H/2 - 100, H/2 + 100]
    bounds = [
        (-500.0, 500.0),
        (-500.0, 500.0),
        (alt0 - 150.0, alt0 - 10.0),
        (0.5 * f_init, 2.0 * f_init),
        (-2.0 * np.pi, 2.0 * np.pi),
        (W/2.0 - 100.0, W/2.0 + 100.0),
        (H/2.0 - 100.0, H/2.0 + 100.0)
    ]
    
    res = opt.minimize(loss_function, best_init, args=(data_points,), bounds=bounds, method='L-BFGS-B')
    
    if res.success:
        N_f, E_f, alt_g, f_val, phi, x_0, y_0 = res.x
        d_f = alt0 - alt_g
        flag_lat, flag_lon, flag_alt = pm.ned2geodetic(N_f, E_f, d_f, lat0, lon0, alt0)
        rmse = np.sqrt(res.fun / len(data_points))
        
        print(f"\n=== Optimization Succeeded ===")
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
        
        return {
            'lat': flag_lat,
            'lon': flag_lon,
            'alt': flag_alt,
            'rmse': rmse,
            'alt_g': alt_g,
            'phi': phi,
            'f_val': f_val
        }
    else:
        print("Optimization failed.")
        return None

if __name__ == '__main__':
    # Try GL014208.LRV
    fit_for_video('GL014208.LRV', r'C:\Users\mido\Documents\antigravity\focused-babbage\meta_data\GX014208_1_GPS9.json')
    # Try GX014209.MP4
    fit_for_video('GX014209.MP4', r'C:\Users\mido\Documents\antigravity\focused-babbage\meta_data\GX014209_1_GPS9.json')
