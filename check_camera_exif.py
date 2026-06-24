import os
from PIL import Image
from PIL.ExifTags import TAGS

def print_camera_exif():
    path = r'c:\Users\mido\Downloads\frames-20260515T124041Z-3-001\frames\frame_0000.jpg'
    if not os.path.exists(path):
        print("File not found")
        return
    img = Image.open(path)
    exif = img._getexif()
    if exif:
        print("Camera Exif Tags:")
        for k, v in exif.items():
            name = TAGS.get(k, k)
            if 'Focal' in str(name) or 'Resolution' in str(name) or 'Lens' in str(name) or 'Model' in str(name) or 'Make' in str(name) or 'Sensor' in str(name):
                print(f"  {name} (Tag {k}): {v}")
    else:
        print("No EXIF found")

if __name__ == '__main__':
    print_camera_exif()
