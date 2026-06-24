import os
import csv
import json
from telemetrik import extract_all_telemetry

def extract_to_csv(video_path, output_gps_path, output_cori_path, output_grav_path):
    print(f"\nExtracting telemetry from {video_path}...")
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        return
        
    try:
        streams = extract_all_telemetry(video_path)
        
        # 1. Extract GPS (we can use the JSON file or extract from video if possible,
        # but wait, telemetrik doesn't support GPS9 natively! That's why GPS5 wasn't found.
        # But wait! We already have the GPS9 data extracted in the JSON files:
        # meta_data/GX014208_1_GPS9.json for GL014208.LRV (since 4208 is the same flight)
        # meta_data/GX014209_1_GPS9.json for GX014209.MP4.
        # So we can just load GPS from the JSON files and CORI/GRAV from the video using telemetrik!)
        
        # Let's extract CORI
        if "CORI" in streams:
            cori_stream = streams["CORI"]
            print(f"Saving {len(cori_stream.data)} CORI samples...")
            with open(output_cori_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['cts_ms', 'qw', 'qx', 'qy', 'qz'])
                for ts, val in cori_stream.data:
                    writer.writerow([ts, val[0], val[1], val[2], val[3]])
        else:
            print("CORI not found in GPMF")
            
        # Let's extract GRAV
        if "GRAV" in streams:
            grav_stream = streams["GRAV"]
            print(f"Saving {len(grav_stream.data)} GRAV samples...")
            with open(output_grav_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['cts_ms', 'gx', 'gy', 'gz'])
                for ts, val in grav_stream.data:
                    writer.writerow([ts, val[0], val[1], val[2]])
        else:
            print("GRAV not found in GPMF")
            
    except Exception as e:
        print(f"Error extracting from {video_path}: {e}")

def main():
    # 4208
    lrv_path = r'C:\Users\mido\Documents\antigravity\focused-babbage\test_flight\GL014208.LRV'
    extract_to_csv(
        lrv_path,
        None,
        'C:\\Users\\mido\\Documents\\antigravity\\focused-babbage\\GL014208_CORI.csv',
        'C:\\Users\\mido\\Documents\\antigravity\\focused-babbage\\GL014208_GRAV.csv'
    )
    
    # 4209
    mp4_path = r'C:\Users\mido\Documents\antigravity\focused-babbage\test_flight\GX014209.MP4'
    extract_to_csv(
        mp4_path,
        None,
        'C:\\Users\\mido\\Documents\\antigravity\\focused-babbage\\GX014209_CORI.csv',
        'C:\\Users\\mido\\Documents\\antigravity\\focused-babbage\\GX014209_GRAV.csv'
    )

if __name__ == '__main__':
    main()
