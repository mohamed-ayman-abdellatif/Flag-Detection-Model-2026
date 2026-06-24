import os
import urllib.request
import time

FLAG_URLS = {
    "egypt": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Flag_of_Egypt.svg/1280px-Flag_of_Egypt.svg.png",
    "france": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Flag_of_France.svg/1280px-Flag_of_France.svg.png",
    "germany": "https://upload.wikimedia.org/wikipedia/commons/thumb/b/ba/Flag_of_Germany.svg/1280px-Flag_of_Germany.svg.png",
    "korea": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/Flag_of_South_Korea.svg/1280px-Flag_of_South_Korea.svg.png",
    "nato": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/Flag_of_NATO.svg/1280px-Flag_of_NATO.svg.png",
    "japan": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Flag_of_Japan.svg/1280px-Flag_of_Japan.svg.png",
    "italy": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/03/Flag_of_Italy.svg/1280px-Flag_of_Italy.svg.png",
    "us": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/Flag_of_the_United_States.svg/1280px-Flag_of_the_United_States.svg.png",
    "palestine": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/00/Flag_of_Palestine.svg/1280px-Flag_of_Palestine.svg.png",
    "canada": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Flag_of_Canada_%28Pantone%29.svg/1280px-Flag_of_Canada_%28Pantone%29.svg.png",
    "uk": "https://upload.wikimedia.org/wikipedia/en/thumb/a/ae/Flag_of_the_United_Kingdom.svg/1280px-Flag_of_the_United_Kingdom.svg.png",
    "russia": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Flag_of_Russia.svg/1280px-Flag_of_Russia.svg.png"
}

def download_flags():
    dest_dir = r'C:\Users\mido\Documents\antigravity\focused-babbage\flag_templates'
    os.makedirs(dest_dir, exist_ok=True)
    
    print("Downloading high-quality flag images from Wikimedia...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    for name, url in FLAG_URLS.items():
        dest_path = os.path.join(dest_dir, f"{name}.png")
        if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
            print(f"  {name} already downloaded. Skipping.")
            continue
            
        print(f"  Downloading {name} from {url}...")
        
        # Retry loop for rate-limiting
        success = False
        wait_time = 2.0
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req) as response:
                    with open(dest_path, 'wb') as f:
                        f.write(response.read())
                print(f"    Saved to {dest_path}")
                success = True
                break
            except Exception as e:
                print(f"    Attempt {attempt + 1} failed: {e}")
                if "429" in str(e):
                    print(f"    Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    wait_time *= 2.0
                else:
                    # Non-429 error, wait a bit and retry anyway
                    time.sleep(1.0)
        
        if not success:
            print(f"    ERROR: Failed to download {name} after all attempts.")
        
        # Politeness delay between different requests
        time.sleep(1.5)

if __name__ == '__main__':
    download_flags()
