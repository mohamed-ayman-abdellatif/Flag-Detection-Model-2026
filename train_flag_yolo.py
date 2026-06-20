import torch
from ultralytics import YOLO

def main():
    print("=== Training YOLO26 Flag Detector (1-Class: 'flag') ===")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA: {cuda_available}  |  torch: {torch.__version__}")
    device_val = 0 if cuda_available else 'cpu'
    print(f"Device: {device_val}")

    model = YOLO('yolo26n.pt')
    print("Starting 1-class flag detection training (10 epochs, imgsz=320)...")
    model.train(
        data='C:/Users/mido/Documents/antigravity/focused-babbage/synthetic_dataset/dataset_1class.yaml',
        epochs=10,
        imgsz=320,
        batch=32,
        device=device_val,
        workers=4,
        patience=5,
        lr0=0.01,
        lrf=0.01,
        mosaic=1.0,
        mixup=0.1,
        flipud=0.3,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=30.0,
        scale=0.5,
        warmup_epochs=2,
        close_mosaic=5,
        name='flag_detector',
        exist_ok=True,
    )
    print("\nDone! Weights saved in runs/detect/flag_detector/weights/best.pt")

if __name__ == '__main__':
    main()
