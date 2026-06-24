import os
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

def inspect_metadata():
    path = r'c:\Users\mido\Downloads\frames-20260515T124041Z-3-001\frames\frame_0700.jpg'
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
        
    img = Image.open(path)
    exif = img._getexif()

    print('Raw EXIF Tags:')
    if exif:
        for k, v in exif.items():
            name = TAGS.get(k, k)
            if name == 'GPSInfo':
                gps_info = {GPSTAGS.get(gk, gk): gv for gk, gv in v.items()}
                print(f'{name}: {gps_info}')
            else:
                print(f'{name} (Tag {k}): {v}')
    else:
        print('No EXIF found')

    # Scan raw file for interesting ASCII strings
    print('\nScanning raw file for strings:')
    with open(path, 'rb') as f:
        data = f.read()
        
    keywords = [b'yaw', b'pitch', b'roll', b'gimbal', b'flight', b'camera', b'dji', b'speed', b'altitude']
    for kw in keywords:
        pos = 0
        count = 0
        while True:
            pos = data.find(kw, pos)
            if pos == -1:
                break
            start = max(0, pos - 40)
            end = min(len(data), pos + len(kw) + 40)
            context = data[start:end]
            # Decode context as latin1 and represent as a safe string
            context_safe = "".join(c if 32 <= ord(c) < 127 else "?" for c in context.decode("latin-1"))
            print(f'Keyword "{kw.decode()}" found at {pos}:')
            print(f'  Context: ...{context_safe}...')
            pos += len(kw)
            count += 1
            if count >= 3:
                print(f'  (Truncated, found additional occurrences)')
                break

if __name__ == '__main__':
    inspect_metadata()
