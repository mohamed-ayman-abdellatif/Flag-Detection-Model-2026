import os
from telemetrik import extract_all_telemetry

def check_telemetry():
    lrv_path = r'C:\Users\mido\Documents\antigravity\focused-babbage\test_flight\GL014208.LRV'
    mp4_path = r'C:\Users\mido\Documents\antigravity\focused-babbage\test_flight\GX014209.MP4'
    
    # Force target to be the high-resolution MP4 file
    target = mp4_path
    print(f"Reading telemetry from: {target}")
    
    try:
        streams = extract_all_telemetry(target)
        print("Available streams in GPMF:")
        for stream_name in streams.keys():
            print(f"  - {stream_name}")
            
        if "GPS5" in streams:
            gps_stream = streams["GPS5"]
            gps_data = gps_stream.data
            print(f"\nFound GPS5 stream with {len(gps_data)} samples!")
            print("First 5 GPS samples:")
            # Each sample is: (timestamp_ms, (lat, lon, alt, spd2d, spd3d, fix, dop))
            for ts, values in gps_data[:5]:
                print(f"  t={ts}ms: Lat={values[0]}, Lon={values[1]}, Alt={values[2]}m, Fix={values[5]}")
        else:
            print("\nGPS5 stream not found in this file.")
            
    except Exception as e:
        print(f"Error extracting telemetry: {e}")

if __name__ == '__main__':
    check_telemetry()
