import os
import json
import zipfile
import shutil
import subprocess

def main():
    base_dir = r"C:\Users\mido\Documents\antigravity\focused-babbage"
    kaggle_json_path = os.path.expanduser('~/.kaggle/kaggle.json')
    if not os.path.exists(kaggle_json_path):
        print("Kaggle credentials not found.")
        return
        
    with open(kaggle_json_path, 'r') as f:
        creds = json.load(f)
        username = creds.get('username')
        api_token = creds.get('key')
        
    env = os.environ.copy()
    env['KAGGLE_API_TOKEN'] = api_token
    env['KAGGLE_USERNAME'] = username
    
    cli_path = r"C:\Users\mido\AppData\Local\Python\pythoncore-3.14-64\Scripts\kaggle.exe"
    kernel_ref = f"{username}/train-yolo26-small-flag-detector"
    
    output_dir = os.path.join(base_dir, "runs", "kaggle_results_new")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading latest outputs from kernel: {kernel_ref}...")
    # Clean output dir first to avoid skipping
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    # Run the download command
    res = subprocess.run([cli_path, 'kernels', 'output', kernel_ref, '-p', output_dir, '--force'],
                         capture_output=True, env=env)
    
    # Print status safely
    print(f"CLI Return Code: {res.returncode}")
    # Decode safely with replace to avoid CharMap errors
    stdout_str = res.stdout.decode('utf-8', errors='replace')
    stderr_str = res.stderr.decode('utf-8', errors='replace')
    print("Download output:")
    print(stdout_str[:1000])
    if stderr_str:
        print("Download errors/info:")
        print(stderr_str[:1000])
        
    zip_path = os.path.join(output_dir, "yolo26_small_results.zip")
    if os.path.exists(zip_path):
        print("\nZip file found. Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print("Extraction complete!")
        
        best_pt_src = os.path.join(output_dir, "weights", "best.pt")
        best_pt_dst = os.path.join(base_dir, "yolo26s_flag_best.pt")
        
        if os.path.exists(best_pt_src):
            # Backup old weights
            old_backup = os.path.join(base_dir, "yolo26s_flag_best_old.pt")
            if os.path.exists(best_pt_dst):
                shutil.copy2(best_pt_dst, old_backup)
                print(f"Backed up old weights to {old_backup}")
                
            shutil.copy2(best_pt_src, best_pt_dst)
            print(f"SAVED NEW TRAINED WEIGHTS TO: {best_pt_dst}")
            print(f"New weights size: {os.path.getsize(best_pt_dst)} bytes")
        else:
            print(f"Error: best.pt not found at {best_pt_src}")
    else:
        print(f"Error: Zip file not found at {zip_path}")

if __name__ == '__main__':
    main()
