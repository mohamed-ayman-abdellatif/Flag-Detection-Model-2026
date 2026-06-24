import csv
import numpy as np
import pymap3d as pm
import scipy.optimize as opt

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

def loss_function(params, data_points):
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
        total_loss += dist**2
    return total_loss

def run_fit(data_points, initial_params):
    bounds = [
        (-400.0, 400.0),
        (-400.0, 400.0),
        (-100.0, 245.0),
        (1000.0, 6000.0),
        (-np.pi * 2, np.pi * 2),
        (1800.0, 2040.0),
        (1000.0, 1160.0)
    ]
    res = opt.minimize(loss_function, initial_params, args=(data_points,), bounds=bounds, method='L-BFGS-B')
    return res.x if res.success else None

def main():
    telemetry, detections = load_data()
    ned_coords, headings, lat0, lon0, alt0 = compute_headings(telemetry)
    
    pass1_frames = [f"frame_{i:04d}.jpg" for i in range(389, 397) if f"frame_{i:04d}.jpg" in detections]
    pass2_frames = [f"frame_{i:04d}.jpg" for i in range(691, 710) if i != 699 and f"frame_{i:04d}.jpg" in detections]
    
    all_frames = pass1_frames + pass2_frames
    
    def get_points(frames):
        pts = []
        for f in frames:
            t = telemetry[f]
            d = detections[f]
            n, e, _ = ned_coords[f]
            pts.append({
                'frame': f,
                'alt_d': t['alt'],
                'n_d': n,
                'e_d': e,
                'yaw_d': headings[f],
                'x_pixel': d['x'],
                'y_pixel': d['y']
            })
        return pts

    # Initial params from grid search
    init_params = [-39.5656, -24.0165, 73.2123, 6000.00, np.radians(98.10), 2040.00, 1160.00]
    
    # 1. Fit on Pass 1 only
    pts1 = get_points(pass1_frames)
    params1 = run_fit(pts1, init_params)
    
    # 2. Fit on Pass 2 only
    pts2 = get_points(pass2_frames)
    params2 = run_fit(pts2, init_params)
    
    # 3. Fit on All
    pts_all = get_points(all_frames)
    params_all = run_fit(pts_all, init_params)
    
    print("=== Validation Results ===")
    
    for label, params in [("Pass 1 only", params1), ("Pass 2 only", params2), ("Both Passes", params_all)]:
        if params is not None:
            N_f, E_f, alt_g, f, phi, x_0, y_0 = params
            d_f = alt0 - alt_g
            lat, lon, alt = pm.ned2geodetic(N_f, E_f, d_f, lat0, lon0, alt0)
            print(f"{label:12s}: Lat={lat:.8f} N, Lon={lon:.8f} E, Alt={alt:.2f} m | NED=({N_f:.2f}, {E_f:.2f}), alt_g={alt_g:.2f} m, f={f:.1f} px, phi={np.degrees(phi):.1f} deg")

if __name__ == '__main__':
    main()
