import csv
import pymap3d as pm

def analyze_all_detections():
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
                
    detections = []
    with open('flag_detections.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['detected'] == '1':
                detections.append({
                    'frame': row['frame'],
                    'x': float(row['x']),
                    'y': float(row['y']),
                    'area': float(row['area'])
                })
                
    # Use first telemetry frame as origin
    first_frame = sorted(telemetry.keys())[0]
    lat0 = telemetry[first_frame]['lat']
    lon0 = telemetry[first_frame]['lon']
    alt0 = telemetry[first_frame]['alt']
    
    print("Frame | Drone N (m) | Drone E (m) | Drone Alt (m) | Flag X | Flag Y | Area")
    print("-" * 80)
    for det in detections:
        frame = det['frame']
        if frame in telemetry:
            t = telemetry[frame]
            n, e, d = pm.geodetic2ned(t['lat'], t['lon'], t['alt'], lat0, lon0, alt0)
            print(f"{frame} | {n:11.2f} | {e:11.2f} | {t['alt']:13.2f} | {det['x']:6.1f} | {det['y']:6.1f} | {det['area']:6.1f}")

if __name__ == '__main__':
    analyze_all_detections()
