import csv
import numpy as np
import scipy.optimize as opt
import pymap3d as pm

def load_data():
    telemetry = {}
    with open('drone_telemetry.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['lat'] and row['lon'] and row['alt']:
                telemetry[row['frame']] = {
                    'lat': float(row['lat']),
                    'lon': float(row['lon']),
                    'alt': float(row['alt'])
                }
                
    detections = {}
    with open('flag_detections.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['detected'] == '1':
                detections[row['frame']] = {
                    'x': float(row['x']),
                    'y': float(row['y']),
                    'area': float(row['area'])
                }
    return telemetry, detections

def compute_headings(telemetry):
    # Sort frames to compute velocities in order
    sorted_frames = sorted(telemetry.keys())
    
    # Origin (Frame 0)
    lat0 = telemetry[sorted_frames[0]]['lat']
    lon0 = telemetry[sorted_frames[0]]['lon']
    alt0 = telemetry[sorted_frames[0]]['alt']
    
    ned_coords = {}
    for frame in sorted_frames:
        t = telemetry[frame]
        n, e, d = pm.geodetic2ned(t['lat'], t['lon'], t['alt'], lat0, lon0, alt0)
        ned_coords[frame] = (n, e, d)
        
    headings = {}
    n_frames = len(sorted_frames)
    
    for idx, frame in enumerate(sorted_frames):
        # We can estimate heading from velocity vector
        # Let's look forward/backward by a few frames to smooth out noise
        start_idx = max(0, idx - 2)
        end_idx = min(n_frames - 1, idx + 2)
        
        n_start, e_start, _ = ned_coords[sorted_frames[start_idx]]
        n_end, e_end, _ = ned_coords[sorted_frames[end_idx]]
        
        dn = n_end - n_start
        de = e_end - e_start
        
        if np.sqrt(dn**2 + de**2) > 0.05: # if drone has moved
            heading = np.atan2(de, dn) # Radians clockwise from North
        else:
            # If not moving (or very slow), copy nearest available heading
            heading = 0.0
            
        headings[frame] = heading
        
    # Fill in static values if any
    last_heading = 0.0
    for frame in sorted_frames:
        if headings[frame] == 0.0:
            headings[frame] = last_heading
        else:
            last_heading = headings[frame]
            
    return ned_coords, headings

def loss_function(params, data_points):
    # params: [N_f, E_f, alt_g, f, phi, x_0, y_0]
    # phi is the camera yaw offset relative to the drone's heading
    N_f, E_f, alt_g, f, phi, x_0, y_0 = params
    
    total_loss = 0
    for dp in data_points:
        N_d = dp['n_d']
        E_d = dp['e_d']
        alt_d = dp['alt_d']
        yaw_d = dp['yaw_d']
        x_meas = dp['x_pixel']
        y_meas = dp['y_pixel']
        
        # Camera yaw in the world
        theta = yaw_d + phi
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        h = alt_d - alt_g
        if h <= 0.1:
            return 1e12
            
        dy = N_f - N_d
        dx = E_f - E_d
        
        # Projection w.r.t rotated camera frame
        # Camera X points right (aligned with rotated East)
        # Camera Y points down (aligned with rotated South)
        X_c = dx * cos_t - dy * sin_t
        Y_c = -dx * sin_t - dy * cos_t
        
        x_proj = x_0 + f * X_c / h
        y_proj = y_0 + f * Y_c / h
        
        dist = np.sqrt((x_proj - x_meas)**2 + (y_proj - y_meas)**2)
        
        # Huber Loss
        delta = 20.0 # pixels
        if dist < delta:
            loss = 0.5 * (dist**2)
        else:
            loss = delta * (dist - 0.5 * delta)
        total_loss += loss
        
    return total_loss

def fit():
    telemetry, detections = load_data()
    ned_coords, headings = compute_headings(telemetry)
    
    first_frame = sorted(telemetry.keys())[0]
    lat0 = telemetry[first_frame]['lat']
    lon0 = telemetry[first_frame]['lon']
    alt0 = telemetry[first_frame]['alt']
    
    common_frames = sorted(set(telemetry.keys()) & set(detections.keys()))
    
    data_points = []
    for frame in common_frames:
        t = telemetry[frame]
        d = detections[frame]
        n, e, d_coord = ned_coords[frame]
        yaw_d = headings[frame]
        data_points.append({
            'frame': frame,
            'alt_d': t['alt'],
            'n_d': n,
            'e_d': e,
            'yaw_d': yaw_d,
            'x_pixel': d['x'],
            'y_pixel': d['y'],
            'area': d['area']
        })
        
    print(f"Total detections: {len(data_points)}")
    
    # Try fitting on ALL detections first
    # Bounds:
    # N_f: [-400, 400], E_f: [-400, 400]
    # alt_g: [0, 245]
    # f: [1000, 6000]
    # phi: [-pi, pi]
    bounds = [
        (-400.0, 400.0),   # N_f
        (-400.0, 400.0),   # E_f
        (-100.0, 245.0),   # alt_g
        (1000.0, 6000.0),  # f
        (-np.pi, np.pi),   # phi
        (1850.0, 1990.0),  # x_0
        (1020.0, 1140.0)   # y_0
    ]
    
    initial_guess = [-6.0, -20.0, 180.0, 3000.0, 0.0, 1920.0, 1080.0]
    
    res = opt.minimize(loss_function, initial_guess, args=(data_points,), bounds=bounds, method='L-BFGS-B')
    
    if res.success:
        N_f, E_f, alt_g, f, phi, x_0, y_0 = res.x
        print("\n=== Fitting with Dynamic Drone Yaw Succeeded ===")
        print(f"Optimal Local Parameters (origin = frame_0000):")
        print(f"  Flag NED North:  {N_f:.4f} m")
        print(f"  Flag NED East:   {E_f:.4f} m")
        print(f"  Ground Altitude: {alt_g:.4f} m")
        print(f"  Focal Length:    {f:.2f} px")
        print(f"  Gimbal Offset (phi): {np.degrees(phi):.2f} degrees")
        print(f"  Principal Point: ({x_0:.2f}, {y_0:.2f})")
        
        # Calculate flag geodetic location
        d_f = alt0 - alt_g
        flag_lat, flag_lon, flag_alt = pm.ned2geodetic(N_f, E_f, d_f, lat0, lon0, alt0)
        print(f"\nFlag Estimated Geodetic Location:")
        print(f"  Latitude:  {flag_lat:.8f} N")
        print(f"  Longitude: {flag_lon:.8f} E")
        print(f"  Altitude:  {flag_alt:.2f} m")
        
        # Compute reprojection stats
        inliers = 0
        outliers = 0
        sq_errors = []
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
            
            X_c = dx * cos_t - dy * sin_t
            Y_c = -dx * sin_t - dy * cos_t
            
            x_proj = x_0 + f * X_c / h
            y_proj = y_0 + f * Y_c / h
            
            dist = np.sqrt((x_proj - x_meas)**2 + (y_proj - y_meas)**2)
            sq_errors.append(dist)
            
            if dist < 30.0:
                inliers += 1
            else:
                outliers += 1
                
        print(f"Inliers (err < 30px): {inliers} / {len(data_points)} ({inliers/len(data_points)*100:.1f}%)")
        print(f"RMSE: {np.sqrt(np.mean(np.array(sq_errors)**2)):.2f} px")
    else:
        print("Optimization failed.")

if __name__ == '__main__':
    fit()
