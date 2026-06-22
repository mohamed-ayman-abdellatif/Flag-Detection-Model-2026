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

def zip_directory(src_dir, zip_filepath):
    print(f"Zipping {src_dir} into {zip_filepath}...")
    with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                filepath = os.path.join(root, file)
                relpath = os.path.relpath(filepath, src_dir)
                zipf.write(filepath, relpath)
    print("Zipping complete!")

def main():
    print("=== Kaggle YOLO26 Training Orchestrator (Resized Single Upload) ===")
    
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
        print("[ERROR] Kaggle credentials (username/token) not found.")
        sys.exit(1)
        
    # Configure environment
    env = os.environ.copy()
    env['KAGGLE_API_TOKEN'] = api_token
    env['KAGGLE_USERNAME'] = username
    
    # Test connection
    res = run_cli_cmd(['datasets', 'list', '--user', username], env)
    if res.returncode != 0:
        print(f"[ERROR] Failed to authenticate with Kaggle CLI: {res.stderr}")
        sys.exit(1)
    print(f"Authenticated successfully as user: {username}")
    
    base_dir = r"C:\Users\mido\Documents\antigravity\focused-babbage"
    dataset_src = os.path.join(base_dir, "kaggle_dataset_resized")
    temp_dataset_dir = os.path.join(base_dir, "kaggle_dataset_temp")
    temp_kernel_dir = os.path.join(base_dir, "kaggle_kernel_temp")
    
    # 2. Zip and upload dataset
    os.makedirs(temp_dataset_dir, exist_ok=True)
    zip_filepath = os.path.join(temp_dataset_dir, "dataset.zip")
    
    # Zip the resized dataset
    zip_directory(dataset_src, zip_filepath)
    
    # Write dataset metadata
    dataset_slug = "drone-flag-dataset-2026-resized"
    dataset_ref = f"{username}/{dataset_slug}"
    dataset_metadata = {
        "title": "Drone Flag Dataset 2026 Resized",
        "id": dataset_ref,
        "licenses": [{"name": "CC0-1.0"}]
    }
    
    with open(os.path.join(temp_dataset_dir, "dataset-metadata.json"), "w") as f:
        json.dump(dataset_metadata, f, indent=4)
        
    # Check if dataset exists
    status_res = run_cli_cmd(['datasets', 'status', dataset_ref], env)
    exists = status_res.returncode == 0
    
    if exists:
        print(f"Dataset {dataset_ref} exists on Kaggle. Uploading a new version...")
        res = run_cli_cmd(['datasets', 'version', '-p', temp_dataset_dir, '-m', "Update resized dataset"], env)
    else:
        print(f"Creating new dataset {dataset_ref} on Kaggle...")
        res = run_cli_cmd(['datasets', 'create', '-p', temp_dataset_dir], env)
        
    print("STDOUT:", res.stdout)
    print("STDERR:", res.stderr)
    
    if res.returncode != 0:
        print("[ERROR] Failed to upload dataset to Kaggle.")
        sys.exit(1)
    print("Dataset successfully uploaded!")
    
    # Wait for dataset to be ready/processed on Kaggle
    print("Waiting for dataset to be ready/processed on Kaggle...")
    while True:
        status_res = run_cli_cmd(['datasets', 'status', dataset_ref], env)
        if status_res.returncode == 0:
            status = status_res.stdout.strip()
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Dataset Status: {status}")
            if status == "ready":
                break
        else:
            print(f"Error checking dataset status: {status_res.stderr}")
        time.sleep(15)
        
    # Clean up zipped dataset to save space
    try:
        os.remove(zip_filepath)
    except Exception:
        pass

    # 3. Prepare and push Kaggle Kernel (Notebook)
    print("\nPreparing Kaggle kernel metadata...")
    os.makedirs(temp_kernel_dir, exist_ok=True)
    
    src_notebook = os.path.join(base_dir, "train_yolo26_kaggle.ipynb")
    dst_notebook = os.path.join(temp_kernel_dir, "train_yolo26_kaggle.ipynb")
    shutil.copy2(src_notebook, dst_notebook)
    
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
    
    # 4. Monitor Kernel Run
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
                
                # Normalize to lowercase for comparisons
                status_lower = status.lower()
                    
                if status != last_status:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Status: {status.upper()}")
                    last_status = status
                
                # Check for completion/error/cancel
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
        
    # 5. Retrieve Results
    if last_status and any(term in last_status.lower() for term in ['complete', 'success']):
        results_local_dir = os.path.join(base_dir, "runs", "kaggle_results")
        os.makedirs(results_local_dir, exist_ok=True)
        print(f"\nDownloading kernel outputs to {results_local_dir}...")
        res = run_cli_cmd(['kernels', 'output', kernel_ref, '-p', results_local_dir], env)
        print("STDOUT:", res.stdout)
        print("STDERR:", res.stderr)
        
        if res.returncode == 0:
            print("Results downloaded successfully!")
            zip_results = os.path.join(results_local_dir, "yolo26_small_results.zip")
            if os.path.exists(zip_results):
                print("Extracting training weights and logs...")
                with zipfile.ZipFile(zip_results, 'r') as zip_ref:
                    zip_ref.extractall(results_local_dir)
                print(f"Weights extracted to {results_local_dir}/weights/")
                
                best_pt_src = os.path.join(results_local_dir, "weights", "best.pt")
                best_pt_dst = os.path.join(base_dir, "yolo26s_flag_best.pt")
                if os.path.exists(best_pt_src):
                    shutil.copy2(best_pt_src, best_pt_dst)
                    print(f"Saved model weights to: {best_pt_dst}")
            print("\nAll tasks completed successfully!")
        else:
            print("[ERROR] Failed to download kernel outputs.")
    else:
        print("\nTraining did not complete successfully. Results could not be downloaded.")
        sys.exit(1)

if __name__ == "__main__":
    main()
