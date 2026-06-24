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

def loss_function_fixed_f(params, f_val, data_points):
    # params: [N_f, E_f, alt_g, phi, x_0, y_0]
    N_f, E_f, alt_g, phi, x_0, y_0 = params
    
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
        
        x_proj = x_0 + f_val * X_c / h
        y_proj = y_0 + f_val * Y_c / h
        
        dist = np.sqrt((x_proj - x_meas)**2 + (y_proj - y_meas)**2)
        total_loss += dist**2
    return total_loss

def main():
    telemetry, detections = load_data()
    ned_coords, headings, lat0, lon0, alt0 = compute_headings(telemetry)
    
    clean_frames = []
    for i in range(389, 397):
        clean_frames.append(f"frame_{i:04d}.jpg")
    for i in range(691, 710):
        if i != 699:
            clean_frames.append(f"frame_{i:04d}.jpg")
            
    data_points = []
    for frame in clean_frames:
        if frame in telemetry and frame in detections:
            t = telemetry[frame]
            d = detections[frame]
            n, e, d_coord = ned_coords[frame]
            data_points.append({
                'frame': frame,
                'alt_d': t['alt'],
                'n_d': n,
                'e_d': e,
                'yaw_d': headings[frame],
                'x_pixel': d['x'],
                'y_pixel': d['y']
            })
            
    print("=== Testing Different Fixed Focal Lengths ===")
    
    # We will test f = 2500, 3000, 3500, 4000, 5000, 6000
    for f_val in [2500.0, 3000.0, 3500.0, 4000.0, 5000.0, 6000.0]:
        # Perform grid search to initialize for this f
        best_loss = 1e9
        best_init = None
        for alt_g_val in np.linspace(0.0, 240.0, 49):
            for phi_deg in np.linspace(-180.0, 180.0, 37):
                phi_val = np.radians(phi_deg)
                A_rows = []
                B_rows = []
                for dp in data_points:
                    h = dp['alt_d'] - alt_g_val
                    theta = dp['yaw_d'] + phi_val
                    cos_t = np.cos(theta)
                    sin_t = np.sin(theta)
                    A_rows.append([cos_t, -sin_t])
                    B_rows.append((dp['x_pixel'] - 1920.0) * h / f_val + dp['e_d'] * cos_t - dp['n_d'] * sin_t)
                    A_rows.append([-sin_t, -cos_t])
                    B_rows.append((dp['y_pixel'] - 1080.0) * h / f_val - dp['e_d'] * sin_t - dp['n_d'] * cos_t)
                xy_f, _, _, _ = np.linalg.lstsq(np.array(A_rows), np.array(B_rows), rcond=None)
                loss = loss_function_fixed_f([xy_f[1], xy_f[0], alt_g_val, phi_val, 1920.0, 1080.0], f_val, data_points)
                if loss < best_loss:
                    best_loss = loss
                    best_init = [xy_f[1], xy_f[0], alt_g_val, phi_val, 1920.0, 1080.0]
                    
        # Optimize
        bounds = [
            (-400.0, 400.0),
            (-400.0, 400.0),
            (-100.0, 245.0),
            (-np.pi * 2, np.pi * 2),
            (1800.0, 2040.0),
            (1000.0, 1160.0)
        ]
        res = opt.minimize(loss_function_fixed_f, best_init, args=(f_val, data_points), bounds=bounds, method='L-BFGS-B')
        if res.success:
            N_f, E_f, alt_g, phi, x_0, y_0 = res.x
            d_f = alt0 - alt_g
            lat, lon, alt = pm.ned2geodetic(N_f, E_f, d_f, lat0, lon0, alt0)
            rmse = np.sqrt(res.fun / len(data_points))
            print(f"f={f_val:.0f} px: Lat={lat:.8f} N, Lon={lon:.8f} E, Alt={alt_g:.2f} m | NED=({N_f:.2f}, {E_f:.2f}), phi={np.degrees(phi):.1f} deg, RMSE={rmse:.2f} px")

if __name__ == '__main__':
    main()
