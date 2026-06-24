import os
import json
import subprocess

def main():
    kaggle_json_path = os.path.expanduser('~/.kaggle/kaggle.json')
    if not os.path.exists(kaggle_json_path):
        print("Kaggle JSON not found.")
        return
        
    with open(kaggle_json_path, 'r') as f:
        creds = json.load(f)
        username = creds.get('username')
        api_token = creds.get('key')
        
    env = os.environ.copy()
    env['KAGGLE_API_TOKEN'] = api_token
    env['KAGGLE_USERNAME'] = username
    
    cli_path = r"C:\Users\mido\AppData\Local\Python\pythoncore-3.14-64\Scripts\kaggle.exe"
    
    # Check dataset status
    res = subprocess.run([cli_path, 'datasets', 'status', f"{username}/drone-flag-dataset-2026-resized"],
                         capture_output=True, text=True, env=env)
    print("Dataset Status:")
    print("STDOUT:", res.stdout.strip())
    print("STDERR:", res.stderr.strip())
    
    # Check kernel status
    res2 = subprocess.run([cli_path, 'kernels', 'status', f"{username}/train-yolo26-small-flag-detector"],
                          capture_output=True, text=True, env=env)
    print("\nKernel Status:")
    print("STDOUT:", res2.stdout.strip())
    print("STDERR:", res2.stderr.strip())
    
    # Check kernel logs
    res3 = subprocess.run([cli_path, 'kernels', 'logs', f"{username}/train-yolo26-small-flag-detector"],
                          capture_output=True, text=True, env=env)
    print("\nKernel Logs:")
    logs_out = res3.stdout.strip()
    if len(logs_out) > 2000:
        logs_out = "[LOGS TRUNCATED FOR BREVITY...]\n" + logs_out[-2000:]
    print(logs_out)
    if res3.stderr.strip():
        print("STDERR:", res3.stderr.strip())

if __name__ == '__main__':
    main()
