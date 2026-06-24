import os
import glob
import csv
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def get_gps_data(exif_data):
    if not exif_data:
        return None
    
    gps_info = {}
    for k, v in exif_data.items():
        name = TAGS.get(k, k)
        if name == 'GPSInfo':
            for gk, gv in v.items():
                tag = GPSTAGS.get(gk, gk)
                gps_info[tag] = gv
    return gps_info

def convert_to_degrees(value):
    d = float(value[0])
    m = float(value[1])
    s = float(value[2])
    return d + (m / 60.0) + (s / 3600.0)

def extract_all():
    frames_dir = r'c:\Users\mido\Downloads\frames-20260515T124041Z-3-001\frames'
    img_paths = sorted(glob.glob(os.path.join(frames_dir, 'frame_*.jpg')))
    
    records = []
    for path in img_paths:
        filename = os.path.basename(path)
        try:
            img = Image.open(path)
            exif = img._getexif()
            gps = get_gps_data(exif)
            if gps:
                lat_ref = gps.get('GPSLatitudeRef', 'N')
                lat_raw = gps.get('GPSLatitude')
                lon_ref = gps.get('GPSLongitudeRef', 'E')
                lon_raw = gps.get('GPSLongitude')
                alt = gps.get('GPSAltitude')
                
                lat = convert_to_degrees(lat_raw)
                if lat_ref != 'N':
                    lat = -lat
                    
                lon = convert_to_degrees(lon_raw)
                if lon_ref != 'E':
                    lon = -lon
                
                alt_val = float(alt)
                
                records.append({
                    'frame': filename,
                    'lat': lat,
                    'lon': lon,
                    'alt': alt_val
                })
            else:
                records.append({
                    'frame': filename,
                    'lat': None,
                    'lon': None,
                    'alt': None
                })
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            
    csv_path = 'drone_telemetry.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame', 'lat', 'lon', 'alt'])
        writer.writeheader()
        writer.writerows(records)
        
    print(f"Extracted telemetry for {len(records)} frames and saved to {csv_path}")
    print("First 5 records:")
    for r in records[:5]:
        print(r)
    print("Last 5 records:")
    for r in records[-5:]:
        print(r)

if __name__ == '__main__':
    extract_all()
