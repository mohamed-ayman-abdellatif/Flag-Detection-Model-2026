import os
from telemetrik import extract_all_telemetry

def check_cori():
    mp4_path = r'C:\Users\mido\Documents\antigravity\focused-babbage\test_flight\GX014209.MP4'
    print(f"Reading CORI from: {mp4_path}")
    
    try:
        streams = extract_all_telemetry(mp4_path)
        if "CORI" in streams:
            cori_stream = streams["CORI"]
            print(f"CORI stream name: {cori_stream.name}")
            print(f"CORI stream samples: {len(cori_stream.data)}")
            print("First 5 samples:")
            for ts, val in cori_stream.data[:5]:
                print(f"  t={ts}ms: {val}")
        else:
            print("CORI stream not found")
            
        if "GRAV" in streams:
            grav_stream = streams["GRAV"]
            print(f"\nGRAV stream samples: {len(grav_stream.data)}")
            print("First 5 samples:")
            for ts, val in grav_stream.data[:5]:
                print(f"  t={ts}ms: {val}")
        else:
            print("GRAV stream not found")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    check_cori()
