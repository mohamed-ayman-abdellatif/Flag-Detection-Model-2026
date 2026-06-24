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

def loss_function(params, data_points, W, H):
    # params: [N_f, E_f, alt_g, f, theta, x_0, y_0]
    N_f, E_f, alt_g, f, theta, x_0, y_0 = params
    
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    
    total_loss = 0
    errors = []
    
    for dp in data_points:
        N_d = dp['n_d']
        E_d = dp['e_d']
        alt_d = dp['alt_d']
        x_meas = dp['x_pixel']
        y_meas = dp['y_pixel']
        
        # Height of drone above ground
        h = alt_d - alt_g
        if h <= 0.1: # prevent division by zero or negative heights
            return 1e10
            
        # Relative coordinates of flag w.r.t. drone
        dy = N_f - N_d # North
        dx = E_f - E_d # East
        
        # Rotate from NED to camera coordinates
        # Camera X points right (aligned with rotated East)
        # Camera Y points down (aligned with rotated South)
        # Assuming camera points straight down (nadir)
        X_c = cos_t * dx + sin_t * dy
        Y_c = -sin_t * dx + cos_t * dy
        
        # Perspective projection
        x_proj = x_0 + f * X_c / h
        y_proj = y_0 + f * Y_c / h
        
        # Errors
        ex = x_proj - x_meas
        ey = y_proj - y_meas
        dist = np.sqrt(ex**2 + ey**2)
        
        # Huber loss (robust to outliers)
        delta = 20.0 # pixels threshold
        if dist < delta:
            loss = 0.5 * (dist**2)
        else:
            loss = delta * (dist - 0.5 * delta)
            
        total_loss += loss
        errors.append(dist)
        
    return total_loss

def fit():
    telemetry, detections = load_data()
    
    # Origin (Frame 0)
    first_frame = sorted(telemetry.keys())[0]
    lat0 = telemetry[first_frame]['lat']
    lon0 = telemetry[first_frame]['lon']
    alt0 = telemetry[first_frame]['alt']
    
    common_frames = sorted(set(telemetry.keys()) & set(detections.keys()))
    
    data_points = []
    for frame in common_frames:
        t = telemetry[frame]
        d = detections[frame]
        n, e, _ = pm.geodetic2ned(t['lat'], t['lon'], t['alt'], lat0, lon0, alt0)
        data_points.append({
            'frame': frame,
            'alt_d': t['alt'],
            'n_d': n,
            'e_d': e,
            'x_pixel': d['x'],
            'y_pixel': d['y']
        })
        
    W = 3840.0
    H = 2160.0
    
    # Initial guess:
    # Let's assume ground altitude is around 100 meters (or 200m). Let's guess alt_g = 180 m
    # Let's assume focal length is around 3000 pixels
    # Let's assume theta = 0 (aligned with axes)
    # Principal point is center of image: x_0 = 1920, y_0 = 1080
    # For N_f and E_f, let's use the drone position where the flag is near the center of the image.
    # In frame_0700, the flag is at (2862, 1690), which is close to center. The drone is at n = -6.16, e = -21.75
    init_N_f = -6.0
    init_E_f = -20.0
    init_alt_g = 180.0
    init_f = 3000.0
    init_theta = 0.0
    init_x0 = 1920.0
    init_y0 = 1080.0
    
    initial_guess = [init_N_f, init_E_f, init_alt_g, init_f, init_theta, init_x0, init_y0]
    
    # Define bounds to keep optimization stable
    # N_f: [-400, 400]
    # E_f: [-400, 400]
    # alt_g: [0, 240] (must be less than minimum drone altitude 251.25)
    # f: [1000, 6000]
    # theta: [-np.pi, np.pi]
    # x_0, y_0: close to center
    bounds = [
        (-400.0, 400.0), # N_f
        (-400.0, 400.0), # E_f
        (-100.0, 245.0),  # alt_g (takeoff altitude is 251.25, so ground must be below it)
        (1000.0, 6000.0), # f
        (-np.pi, np.pi),  # theta
        (1800.0, 2040.0), # x_0
        (1000.0, 1160.0)  # y_0
    ]
    
    res = opt.minimize(loss_function, initial_guess, args=(data_points, W, H), bounds=bounds, method='L-BFGS-B')
    
    if res.success:
        print("Optimization Succeeded!")
        N_f, E_f, alt_g, f, theta, x_0, y_0 = res.x
        print(f"Fit Parameters:")
        print(f"  Flag NED North: {N_f:.4f} m")
        print(f"  Flag NED East:  {E_f:.4f} m")
        print(f"  Ground Altitude: {alt_g:.4f} m")
        print(f"  Focal Length:   {f:.2f} pixels")
        print(f"  Camera Yaw:     {np.degrees(theta):.2f} degrees")
        print(f"  Principal Point: ({x_0:.2f}, {y_0:.2f})")
        
        # Convert flag NED back to geodetic GPS coordinates
        flag_lat, flag_lon, flag_alt = pm.ned2geodetic(N_f, E_f, -alt_g, lat0, lon0, alt0)
        # Note: the Down coordinate of the flag on the ground is -alt_g
        # wait! alt_g is altitude above sea level, and alt0 is drone altitude above sea level.
        # The NED coordinate system has Down pointing downwards, so D = -(alt_g - alt0).
        # Let's check: pm.ned2geodetic(n, e, d, lat0, lon0, alt0)
        # Here, d = -(alt_g - alt0) = alt0 - alt_g
        d_f = alt0 - alt_g
        flag_lat, flag_lon, flag_alt = pm.ned2geodetic(N_f, E_f, d_f, lat0, lon0, alt0)
        
        print(f"\nFlag Geographic Location:")
        print(f"  Latitude:  {flag_lat:.8f} N")
        print(f"  Longitude: {flag_lon:.8f} E")
        print(f"  Altitude:  {flag_alt:.2f} m")
        
        # Let's analyze errors / inliers
        inliers = 0
        outliers = 0
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        print("\nDetail of top 15 inliers/outliers:")
        for dp in data_points:
            N_d = dp['n_d']
            E_d = dp['e_d']
            alt_d = dp['alt_d']
            x_meas = dp['x_pixel']
            y_meas = dp['y_pixel']
            
            h = alt_d - alt_g
            dy = N_f - N_d
            dx = E_f - E_d
            X_c = cos_t * dx + sin_t * dy
            Y_c = -sin_t * dx + cos_t * dy
            x_proj = x_0 + f * X_c / h
            y_proj = y_0 + f * Y_c / h
            dist = np.sqrt((x_proj - x_meas)**2 + (y_proj - y_meas)**2)
            
            if dist < 30.0:
                inliers += 1
            else:
                outliers += 1
                
        print(f"Total points: {len(data_points)}")
        print(f"Inliers (err < 30 px): {inliers}")
        print(f"Outliers (err >= 30 px): {outliers}")
    else:
        print("Optimization Failed:")
        print(res.message)

if __name__ == '__main__':
    fit()
