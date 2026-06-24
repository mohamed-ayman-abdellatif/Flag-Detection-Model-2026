import csv
import numpy as np
import pymap3d as pm
import random

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
    sorted_frames = sorted(telemetry.keys())
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
        start_idx = max(0, idx - 2)
        end_idx = min(n_frames - 1, idx + 2)
        n_start, e_start, _ = ned_coords[sorted_frames[start_idx]]
        n_end, e_end, _ = ned_coords[sorted_frames[end_idx]]
        dn = n_end - n_start
        de = e_end - e_start
        if np.sqrt(dn**2 + de**2) > 0.05:
            heading = np.atan2(de, dn)
        else:
            heading = 0.0
        headings[frame] = heading
        
    last_heading = 0.0
    for frame in sorted_frames:
        if headings[frame] == 0.0:
            headings[frame] = last_heading
        else:
            last_heading = headings[frame]
            
    return ned_coords, headings, lat0, lon0, alt0

def evaluate(N_f, E_f, alt_g, f, phi, data_points):
    x_0, y_0 = 1920.0, 1080.0
    inliers = 0
    errors = []
    
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
        if h <= 1.0:
            return 0, []
            
        dy = N_f - N_d
        dx = E_f - E_d
        
        X_c = dx * cos_t - dy * sin_t
        Y_c = -dx * sin_t - dy * cos_t
        
        x_proj = x_0 + f * X_c / h
        y_proj = y_0 + f * Y_c / h
        
        dist = np.sqrt((x_proj - x_meas)**2 + (y_proj - y_meas)**2)
        errors.append(dist)
        if dist < 40.0: # threshold in pixels
            inliers += 1
            
    return inliers, errors

def main():
    telemetry, detections = load_data()
    ned_coords, headings, lat0, lon0, alt0 = compute_headings(telemetry)
    
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
        
    print(f"Total data points: {len(data_points)}")
    
    # We will search over a grid
    # Let's search flag N_f and E_f in the range where the drone flies.
    # Drone N range: -246 to 159
    # Drone E range: -133 to 95
    # Let's do a Monte Carlo search first to find the best basin!
    best_inliers = 0
    best_params = None
    best_rmse = 1e9
    
    # Run 100,000 random samples
    np.random.seed(42)
    random.seed(42)
    
    print("Running random search...")
    for step in range(100000):
        N_f = random.uniform(-200.0, 100.0)
        E_f = random.uniform(-100.0, 50.0)
        alt_g = random.uniform(0.0, 240.0)
        f = random.uniform(2000.0, 4500.0)
        phi = random.uniform(-np.pi, np.pi)
        
        inliers, errors = evaluate(N_f, E_f, alt_g, f, phi, data_points)
        if inliers > best_inliers:
            best_inliers = inliers
            best_params = (N_f, E_f, alt_g, f, phi)
            best_rmse = np.sqrt(np.mean(np.array(errors)**2)) if errors else 1e9
            print(f"Step {step}: Inliers={inliers}/{len(data_points)}, RMSE={best_rmse:.2f} px, Params: N={N_f:.2f}, E={E_f:.2f}, Alt_g={alt_g:.2f}, f={f:.2f}, phi={np.degrees(phi):.1f} deg")
            
        elif inliers == best_inliers and inliers > 0:
            rmse = np.sqrt(np.mean(np.array(errors)**2))
            if rmse < best_rmse:
                best_rmse = rmse
                best_params = (N_f, E_f, alt_g, f, phi)
                print(f"Step {step} (lower RMSE): Inliers={inliers}, RMSE={best_rmse:.2f} px, Params: N={N_f:.2f}, E={E_f:.2f}, Alt_g={alt_g:.2f}, f={f:.2f}, phi={np.degrees(phi):.1f} deg")

    if best_params:
        N_f, E_f, alt_g, f, phi = best_params
        print("\nBest parameters found:")
        print(f"  N_f: {N_f:.4f} m")
        print(f"  E_f: {E_f:.4f} m")
        print(f"  Alt_g: {alt_g:.4f} m")
        print(f"  f: {f:.2f} px")
        print(f"  phi: {np.degrees(phi):.2f} deg")
        print(f"  Inliers: {best_inliers} / {len(data_points)}")
        print(f"  RMSE: {best_rmse:.2f} px")
        
        # Convert to geodetic
        d_f = alt0 - alt_g
        flag_lat, flag_lon, flag_alt = pm.ned2geodetic(N_f, E_f, d_f, lat0, lon0, alt0)
        print(f"GPS: Lat={flag_lat:.8f}, Lon={flag_lon:.8f}, Alt={flag_alt:.2f}")
    else:
        print("No solution with inliers found.")

if __name__ == '__main__':
    main()
