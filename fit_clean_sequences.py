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
        
        # Rotated camera frame coordinates
        X_c = dx * cos_t - dy * sin_t
        Y_c = -dx * sin_t - dy * cos_t
        
        x_proj = x_0 + f * X_c / h
        y_proj = y_0 + f * Y_c / h
        
        dist = np.sqrt((x_proj - x_meas)**2 + (y_proj - y_meas)**2)
        total_loss += dist**2
        
    return total_loss

def fit():
    telemetry, detections = load_data()
    ned_coords, headings, lat0, lon0, alt0 = compute_headings(telemetry)
    
    # Define clean frames list
    clean_frames = []
    # Sequence 1: 389 to 396
    for i in range(389, 397):
        clean_frames.append(f"frame_{i:04d}.jpg")
    # Sequence 2: 691 to 709 (excluding 699)
    for i in range(691, 710):
        if i != 699:
            clean_frames.append(f"frame_{i:04d}.jpg")
            
    data_points = []
    for frame in clean_frames:
        if frame in telemetry and frame in detections:
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
                'y_pixel': d['y']
            })
            
    print(f"Number of clean data points: {len(data_points)}")
    
    # Optimize parameters
    # Let's perform a grid search over alt_g and phi to find the global minimum, 
    # since it's a small dataset and L-BFGS-B is sensitive to initial guess.
    best_loss = 1e9
    best_params = None
    
    # Grid search ranges:
    # alt_g: [0, 240] in steps of 5m
    # phi: [-pi, pi] in steps of 10 degrees
    # N_f: [-100, 100], E_f: [-100, 100]
    # f: [2000, 4500] in steps of 250
    # x_0, y_0: fixed to 1920, 1080
    x_0, y_0 = 1920.0, 1080.0
    
    print("Performing coarse grid search on clean data...")
    for alt_g_val in np.linspace(0.0, 240.0, 49): # steps of 5m
        for phi_deg in np.linspace(-180.0, 180.0, 37): # steps of 10 deg
            phi_val = np.radians(phi_deg)
            for f_val in [2500.0, 3000.0, 3500.0, 4000.0]:
                # With alt_g, phi, f fixed, the projection equations are:
                # (x - x_0) * h / f = dx * cos_t - dy * sin_t
                # (y - y_0) * h / f = -dx * sin_t - dy * cos_t
                # Let's solve for N_f and E_f using linear least squares!
                # We can rewrite the equations for each frame:
                # E_f * cos_t - N_f * sin_t = (x - x_0) * h / f + E_d * cos_t - N_d * sin_t
                # -E_f * sin_t - N_f * cos_t = (y - y_0) * h / f - E_d * sin_t - N_d * cos_t
                # Let A_matrix * [E_f, N_f]^T = B_vector
                A_rows = []
                B_rows = []
                for dp in data_points:
                    h = dp['alt_d'] - alt_g_val
                    theta = dp['yaw_d'] + phi_val
                    cos_t = np.cos(theta)
                    sin_t = np.sin(theta)
                    
                    # Row 1 (X_c)
                    A_rows.append([cos_t, -sin_t])
                    B_rows.append((dp['x_pixel'] - x_0) * h / f_val + dp['e_d'] * cos_t - dp['n_d'] * sin_t)
                    
                    # Row 2 (Y_c)
                    A_rows.append([-sin_t, -cos_t])
                    B_rows.append((dp['y_pixel'] - y_0) * h / f_val - dp['e_d'] * sin_t - dp['n_d'] * cos_t)
                    
                A_arr = np.array(A_rows)
                B_arr = np.array(B_rows)
                
                # Solve least squares
                xy_f, _, _, _ = np.linalg.lstsq(A_arr, B_arr, rcond=None)
                E_f_val, N_f_val = xy_f[0], xy_f[1]
                
                # Compute loss
                loss = loss_function([N_f_val, E_f_val, alt_g_val, f_val, phi_val, x_0, y_0], data_points)
                if loss < best_loss:
                    best_loss = loss
                    best_params = [N_f_val, E_f_val, alt_g_val, f_val, phi_val, x_0, y_0]
                    
    print(f"Coarse search best loss: {best_loss:.2f}")
    print(f"Coarse search parameters: N_f={best_params[0]:.2f}, E_f={best_params[1]:.2f}, Alt_g={best_params[2]:.2f}, f={best_params[3]:.2f}, phi={np.degrees(best_params[4]):.2f} deg")
    
    # Step 3: Run gradient-based optimization starting from the best grid search parameters
    bounds = [
        (-400.0, 400.0),   # N_f
        (-400.0, 400.0),   # E_f
        (-100.0, 245.0),   # alt_g
        (1000.0, 6000.0),  # f
        (-np.pi * 2, np.pi * 2),   # phi
        (1800.0, 2040.0),  # x_0
        (1000.0, 1160.0)   # y_0
    ]
    
    res = opt.minimize(loss_function, best_params, args=(data_points,), bounds=bounds, method='L-BFGS-B')
    
    if res.success:
        N_f, E_f, alt_g, f, phi, x_0, y_0 = res.x
        print("\n=== Optimization Succeeded on Clean Data ===")
        print(f"Parameters:")
        print(f"  Flag NED North:  {N_f:.4f} m")
        print(f"  Flag NED East:   {E_f:.4f} m")
        print(f"  Ground Altitude: {alt_g:.4f} m")
        print(f"  Focal Length:    {f:.2f} px")
        print(f"  Gimbal Offset (phi): {np.degrees(phi):.2f} degrees")
        print(f"  Principal Point: ({x_0:.2f}, {y_0:.2f})")
        
        d_f = alt0 - alt_g
        flag_lat, flag_lon, flag_alt = pm.ned2geodetic(N_f, E_f, d_f, lat0, lon0, alt0)
        print(f"\nFlag Estimated Geodetic Location:")
        print(f"  Latitude:  {flag_lat:.8f} N")
        print(f"  Longitude: {flag_lon:.8f} E")
        print(f"  Altitude:  {flag_alt:.2f} m")
        
        # Calculate error stats
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
            dy = N_f - N_d
            dx = E_f - E_d
            
            X_c = dx * cos_t - dy * sin_t
            Y_c = -dx * sin_t - dy * cos_t
            
            x_proj = x_0 + f * X_c / h
            y_proj = y_0 + f * Y_c / h
            
            dist = np.sqrt((x_proj - x_meas)**2 + (y_proj - y_meas)**2)
            errors.append(dist)
            print(f"{dp['frame']} | Meas: ({x_meas:6.1f}, {y_meas:6.1f}) | Proj: ({x_proj:6.1f}, {y_proj:6.1f}) | Error: {dist:6.2f} px")
            
        rmse = np.sqrt(np.mean(np.array(errors)**2))
        print(f"\nRMSE for clean sequence points: {rmse:.2f} px")
        print(f"Max error: {max(errors):.2f} px")
        print(f"Min error: {min(errors):.2f} px")
        
        # Save results
        with open('estimated_flag_location.txt', 'w') as out_f:
            out_f.write("=== OPTIMIZATION RESULT (CLEAN DATA) ===\n")
            out_f.write(f"Flag Latitude:  {flag_lat:.8f} N\n")
            out_f.write(f"Flag Longitude: {flag_lon:.8f} E\n")
            out_f.write(f"Flag Altitude:  {flag_alt:.2f} m\n\n")
            out_f.write("=== ESTIMATED PARAMETERS ===\n")
            out_f.write(f"Flag NED North:  {N_f:.4f} m\n")
            out_f.write(f"Flag NED East:   {E_f:.4f} m\n")
            out_f.write(f"Ground Altitude: {alt_g:.4f} m\n")
            out_f.write(f"Focal Length:    {f:.2f} px\n")
            out_f.write(f"Camera Yaw Offset: {np.degrees(phi):.2f} deg\n")
            out_f.write(f"Principal Point: ({x_0:.2f}, {y_0:.2f})\n")
            out_f.write(f"RMSE Error:      {rmse:.2f} px\n")
    else:
        print("Optimization failed.")

if __name__ == '__main__':
    fit()
