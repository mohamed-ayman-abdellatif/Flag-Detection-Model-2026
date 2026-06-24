import csv
import numpy as np
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

def analyze():
    telemetry, detections = load_data()
    
    # Let's find the origin (use the first telemetry frame)
    first_frame = sorted(telemetry.keys())[0]
    lat0 = telemetry[first_frame]['lat']
    lon0 = telemetry[first_frame]['lon']
    alt0 = telemetry[first_frame]['alt']
    
    print(f"Origin (Frame 0): Lat={lat0:.8f}, Lon={lon0:.8f}, Alt={alt0:.2f} m")
    
    # Collect data points where flag is detected
    common_frames = sorted(set(telemetry.keys()) & set(detections.keys()))
    print(f"Number of frames where flag is detected: {len(common_frames)}")
    
    data_points = []
    for frame in common_frames:
        t = telemetry[frame]
        d = detections[frame]
        
        # Convert GPS to local NED coordinates
        n, e, d_coord = pm.geodetic2ned(t['lat'], t['lon'], t['alt'], lat0, lon0, alt0)
        
        data_points.append({
            'frame': frame,
            'lat_d': t['lat'],
            'lon_d': t['lon'],
            'alt_d': t['alt'],
            'n_d': n,
            'e_d': e,
            'd_d': d_coord,
            'x_pixel': d['x'],
            'y_pixel': d['y']
        })
        
    print("\nSample Data Points (first 10):")
    for dp in data_points[:10]:
        print(f"Frame {dp['frame']}: Drone NED=({dp['n_d']:.2f}, {dp['e_d']:.2f}, {dp['d_d']:.2f}), Pixel=({dp['x_pixel']:.1f}, {dp['y_pixel']:.1f})")

    # Let's check how the drone is moving
    n_coords = [dp['n_d'] for dp in data_points]
    e_coords = [dp['e_d'] for dp in data_points]
    alt_coords = [dp['alt_d'] for dp in data_points]
    
    print("\nDrone trajectory range for detected frames:")
    print(f"  North: {min(n_coords):.2f} to {max(n_coords):.2f} meters (range {max(n_coords)-min(n_coords):.2f} m)")
    print(f"  East:  {min(e_coords):.2f} to {max(e_coords):.2f} meters (range {max(e_coords)-min(e_coords):.2f} m)")
    print(f"  Alt:   {min(alt_coords):.2f} to {max(alt_coords):.2f} meters (range {max(alt_coords)-min(alt_coords):.2f} m)")

if __name__ == '__main__':
    analyze()
