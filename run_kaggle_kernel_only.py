import os
import sys
import json
import shutil
import zipfile
import time
import subprocess

def run_cli_cmd(args, env):
    cli_path = r"C:\Users\mido\AppData\Local\Python\pythoncore-3.14-64\Scripts\kaggle.exe"
    cmd = [cli_path] + args
    res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', env=env)
    return res

def main():
    print("=== Kaggle YOLO26 Kernel-Only Trigger ===")
    
    # 1. Retrieve API key
    kaggle_json_path = os.path.expanduser('~/.kaggle/kaggle.json')
    username = None
    api_token = None
    
    if os.path.exists(kaggle_json_path):
        try:
            with open(kaggle_json_path, 'r') as f:
                creds = json.load(f)
                username = creds.get('username')
                api_token = creds.get('key')
        except Exception:
            pass
            
    if not username:
        username = os.environ.get('KAGGLE_USERNAME')
    if not api_token:
        api_token = os.environ.get('KAGGLE_API_TOKEN') or os.environ.get('KAGGLE_KEY')
        
    if not username or not api_token:
        print("[ERROR] Kaggle credentials not found.")
        sys.exit(1)
        
    env = os.environ.copy()
    env['KAGGLE_API_TOKEN'] = api_token
    env['KAGGLE_USERNAME'] = username
    
    base_dir = r"C:\Users\mido\Documents\antigravity\focused-babbage"
    temp_kernel_dir = os.path.join(base_dir, "kaggle_kernel_temp")
    
    # 2. Prepare and push Kaggle Kernel (Notebook)
    print("\nPreparing Kaggle kernel metadata...")
    os.makedirs(temp_kernel_dir, exist_ok=True)
    
    src_notebook = os.path.join(base_dir, "train_yolo26_kaggle.ipynb")
    dst_notebook = os.path.join(temp_kernel_dir, "train_yolo26_kaggle.ipynb")
    shutil.copy2(src_notebook, dst_notebook)
    
    dataset_ref = f"{username}/drone-flag-dataset-2026-resized"
    kernel_slug = "train-yolo26-small-flag-detector"
    kernel_ref = f"{username}/{kernel_slug}"
    kernel_metadata = {
        "id": kernel_ref,
        "title": "Train YOLO26 Small Flag Detector",
        "code_file": "train_yolo26_kaggle.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [dataset_ref],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }
    
    with open(os.path.join(temp_kernel_dir, "kernel-metadata.json"), "w") as f:
        json.dump(kernel_metadata, f, indent=4)
        
    print(f"Pushing kernel {kernel_ref} to Kaggle...")
    res = run_cli_cmd(['kernels', 'push', '-p', temp_kernel_dir, '--accelerator', 'NvidiaTeslaT4'], env)
    print("STDOUT:", res.stdout)
    print("STDERR:", res.stderr)
    
    if res.returncode != 0:
        print("[ERROR] Failed to push kernel to Kaggle.")
        sys.exit(1)
    print("Kernel successfully pushed! Execution triggered on Kaggle GPU (T4).")
    
    # 3. Monitor Kernel Run
    print("\nMonitoring kernel execution...")
    print("Notebook Link: " + f"https://www.kaggle.com/code/{username}/{kernel_slug}")
    
    last_status = None
    while True:
        try:
            status_res = run_cli_cmd(['kernels', 'status', kernel_ref], env)
            if status_res.returncode == 0:
                output = status_res.stdout.strip()
                status = "unknown"
                if "has status" in output:
                    parts = output.split('"')
                    if len(parts) >= 2:
                        status = parts[1]
                else:
                    status = output
                
                status_lower = status.lower()
                    
                if status != last_status:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Status: {status.upper()}")
                    last_status = status
                
                is_done = False
                for term in ['complete', 'error', 'cancel']:
                    if term in status_lower:
                        is_done = True
                        break
                    
                if is_done:
                    if 'error' in status_lower or 'cancel' in status_lower:
                        print(f"\n[ERROR] Kernel execution failed. Output: {output}")
                    else:
                        print("\n[SUCCESS] Kernel training complete!")
                    break
            else:
                print(f"Error checking status: {status_res.stderr}")
        except Exception as e:
            print(f"Error checking status: {e}")
            
        time.sleep(45)
        
    # 4. Retrieve Results
    if last_status and any(term in last_status.lower() for term in ['complete', 'success']):
        results_local_dir = os.path.join(base_dir, "runs", "kaggle_results")
        os.makedirs(results_local_dir, exist_ok=True)
        print(f"\nDownloading kernel outputs to {results_local_dir}...")
        res = run_cli_cmd(['kernels', 'output', kernel_ref, '-p', results_local_dir], env)
        
        # We always copy weights even if res.returncode is non-zero, as long as files exist
        best_pt_src = os.path.join(results_local_dir, "runs", "detect", "yolo26_small", "weights", "best.pt")
        best_pt_dst = os.path.join(base_dir, "yolo26s_flag_best.pt")
        if os.path.exists(best_pt_src):
            shutil.copy2(best_pt_src, best_pt_dst)
            print(f"Saved model weights to: {best_pt_dst}")
            print("\nAll tasks completed successfully!")
        else:
            # Try fallback to zip extraction
            zip_results = os.path.join(results_local_dir, "yolo26_small_results.zip")
            if os.path.exists(zip_results):
                print("Extracting training weights and logs...")
                with zipfile.ZipFile(zip_results, 'r') as zip_ref:
                    zip_ref.extractall(results_local_dir)
                best_pt_src_extracted = os.path.join(results_local_dir, "weights", "best.pt")
                if os.path.exists(best_pt_src_extracted):
                    shutil.copy2(best_pt_src_extracted, best_pt_dst)
                    print(f"Saved model weights to: {best_pt_dst}")
                    print("\nAll tasks completed successfully!")
                    return
            print("[ERROR] Failed to locate best.pt weight file.")
            sys.exit(1)
    else:
        print("\nTraining did not complete successfully.")
        sys.exit(1)

if __name__ == "__main__":
    main()
