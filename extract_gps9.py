import os
import struct
import pymap3d as pm
from telemetrik.parser import get_boxes, get_samples, get_gpmf_boxes, _from_bytes

def parse_gps9(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
        
    print(f"Extracting GPS9 from: {path}")
    
    with open(path, 'rb') as f:
        file_size = os.path.getsize(path)
        
        # 1. Walk MP4 boxes to get sample table
        minf_boxes = get_boxes(f, 0, file_size, ["moov", "trak", "mdia", "minf"])
        stbl = None
        gpmf_mdia = None
        for box in minf_boxes:
            if get_boxes(f, box.offset, box.size, ["minf", "gmhd", "gpmd"]):
                stbl = get_boxes(f, box.offset, box.size, ["minf", "stbl"])[0]
                mdia_boxes = get_boxes(f, 0, file_size, ["moov", "trak", "mdia"])
                for mdia in mdia_boxes:
                    if box.offset >= mdia.offset and box.offset < mdia.offset + mdia.size:
                        gpmf_mdia = mdia
                        break
                break
                
        if stbl is None:
            print("No GPMF track found")
            return
            
        # 2. Get samples and timescale
        time_base = None
        if gpmf_mdia:
            mdhd_boxes = get_boxes(f, gpmf_mdia.offset, gpmf_mdia.size, ["mdia", "mdhd"])
            if mdhd_boxes:
                mdhd = mdhd_boxes[0]
                f.seek(mdhd.offset + 12)
                f.read(8)  # skip creation/modification times
                timescale = _from_bytes(f.read(4))
                time_base = (1, timescale)
                
        samples = get_samples(f, stbl)
        print(f"Parsed {len(samples)} GPMF samples.")
        
        # 3. Extract GPS9 stream
        records = []
        for sample_idx, sample in enumerate(samples):
            strm_boxes = get_gpmf_boxes(f, sample.offset, sample.size, ["DEVC", "STRM"])
            gps9_strm = None
            for box in strm_boxes:
                if get_gpmf_boxes(f, box.offset, box.size, ["STRM", "GPS9"]):
                    gps9_strm = box
                    break
            if gps9_strm is None:
                continue
                
            # Get timestamps from STMP tag
            stmp_boxes = get_gpmf_boxes(f, gps9_strm.offset, gps9_strm.size, ["STRM", "STMP"])
            if not stmp_boxes:
                continue
            stmp = stmp_boxes[0]
            f.seek(stmp.offset + 8)
            stmp_us = _from_bytes(f.read(stmp.struct_size * stmp.repeat))
            stmp_ms = stmp_us / 1000.0
            
            # Get scales from SCAL tag (contains scale factors for each of the fields)
            # GPS9 fields: Lat, Lon, Alt, Spd2d, Spd3d, Days, Secs, DOP, Fix
            # Typ. scales: 10000000, 10000000, 1000, 100, 100, 1, 1000, 100, 1
            # Let's read SCAL box
            scal_boxes = get_gpmf_boxes(f, gps9_strm.offset, gps9_strm.size, ["STRM", "SCAL"])
            scales = [1.0] * 9
            if scal_boxes:
                scal = scal_boxes[0]
                f.seek(scal.offset + 8)
                # Parse scale factors
                # The scale factors can be 32-bit signed ints
                # Let's read 32-bit ints
                for i in range(scal.repeat):
                    scales[i] = _from_bytes(f.read(4), signed=True)
            
            # Read GPS9 data box
            gps9_box = get_gpmf_boxes(f, gps9_strm.offset, gps9_strm.size, ["STRM", "GPS9"])[0]
            f.seek(gps9_box.offset + 8)
            
            # Since the type is compound 'lllllllSS' (32 bytes per repeat)
            # Let's verify struct_size
            for i in range(gps9_box.repeat):
                raw_data = f.read(32)
                # Unpack big-endian: 7 signed 32-bit integers, 2 unsigned 16-bit integers
                unpacked = struct.unpack(">7i2H", raw_data)
                
                # Apply scales
                lat = unpacked[0] / scales[0]
                lon = unpacked[1] / scales[1]
                alt = unpacked[2] / scales[2]
                spd2d = unpacked[3] / scales[3]
                spd3d = unpacked[4] / scales[4]
                days = unpacked[5]
                secs = unpacked[6] / scales[6]
                dop = unpacked[7] / scales[7]
                fix = unpacked[8]
                
                # Timestamp approximation
                # We can linearly interpolate timestamps if needed, or just use stmp_ms
                records.append({
                    'sample_idx': sample_idx,
                    't_ms': stmp_ms,
                    'lat': lat,
                    'lon': lon,
                    'alt': alt,
                    'spd2d': spd2d,
                    'spd3d': spd3d,
                    'fix': fix,
                    'dop': dop
                })
                
        print(f"Extracted {len(records)} GPS9 records.")
        if records:
            print("First 5 records:")
            for r in records[:5]:
                print(f"  t={r['t_ms']:.1f}ms, Lat={r['lat']:.8f}, Lon={r['lon']:.8f}, Alt={r['alt']:.2f}m, Fix={r['fix']}")
            print("Last 5 records:")
            for r in records[-5:]:
                print(f"  t={r['t_ms']:.1f}ms, Lat={r['lat']:.8f}, Lon={r['lon']:.8f}, Alt={r['alt']:.2f}m, Fix={r['fix']}")
                
if __name__ == '__main__':
    parse_gps9(r'C:\Users\mido\Documents\antigravity\focused-babbage\test_flight\GL014208.LRV')
