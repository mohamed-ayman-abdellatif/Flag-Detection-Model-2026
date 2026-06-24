import os

def parse_mp4_handlers(path):
    if not os.path.exists(path):
        print("File not found")
        return
        
    print(f"Parsing MP4 handlers for: {path}")
    file_size = os.path.getsize(path)
    read_size = min(file_size, 20 * 1024 * 1024)
    with open(path, 'rb') as f:
        f.seek(file_size - read_size)
        data = f.read(read_size)
        
    # Search for 'hdlr' boxes
    pos = 0
    while True:
        pos = data.find(b'hdlr', pos)
        if pos == -1:
            break
            
        # The 'hdlr' box format:
        # 4 bytes size
        # 4 bytes 'hdlr'
        # 4 bytes version/flags (usually 0)
        # 4 bytes predefined (usually 0)
        # 4 bytes handler type (e.g. 'vide', 'soun', 'meta', 'gpmd')
        # 12 bytes manufacturer (usually 0)
        # string name (null-terminated)
        
        # The handler type is at pos + 12
        if pos + 16 <= len(data):
            h_type = data[pos+12:pos+16].decode('latin-1', errors='ignore')
            # Let's read the name (which starts at pos + 28)
            name_pos = pos + 24
            name_end = data.find(b'\x00', name_pos)
            if name_end != -1 and name_end - name_pos < 100:
                name = data[name_pos:name_end].decode('latin-1', errors='ignore')
            else:
                name = "Unknown"
            print(f"Found handler: Type='{h_type}', Name='{name}' at offset {pos}")
            
        pos += 4

if __name__ == '__main__':
    parse_mp4_handlers(r'C:\Users\mido\Documents\antigravity\focused-babbage\test_flight\GX014209.MP4')
