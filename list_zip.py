import zipfile

def list_zip():
    path = r'C:\Users\mido\Documents\antigravity\focused-babbage\test_flight-20260523T211037Z-3-003.zip'
    with zipfile.ZipFile(path, 'r') as zip_ref:
        print("Files inside zip:")
        for name in zip_ref.namelist():
            print(f"  - {name}")

if __name__ == '__main__':
    list_zip()
