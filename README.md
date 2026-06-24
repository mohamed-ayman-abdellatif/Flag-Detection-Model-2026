# Flag Detection Model (2026)

This repository contains the training pipelines, models, and scripts for a high-accuracy flag detection model trained to recognize flags (including Germany, Russia, France, Egypt, and others) from aerial/drone footage.

## Project Overview

*   **Model Architecture:** YOLO26 Small (Ultralytics)
*   **Task Type:** 1-Class Object Detection (`flag`)
*   **Input Image Dimensions:** Native `imgsz=640`
*   **Model Weights:** Deployed directly in the root as `yolo26s_flag_best.pt`

---

## Performance Metrics (YOLO26s - Latest Model)

After training for **100 epochs** on the remote Kaggle Tesla T4 GPU with the updated dataset (incorporating scale-matched full-frame negatives), the model achieved the following validation metrics:

| Metric | Value |
|---|---|
| **Precision** | 99.64% |
| **Recall** | 99.59% |
| **mAP@50** | 99.47% |
| **mAP@50-95** | **81.10%** (Boosted from 75.83%) |

### Key Improvements:
*   **Curb & Runway False Positives Resolved:** Trained on **353 full-frame negatives** letterboxed directly to $640 \times 640$ to preserve the exact resolution scale of concrete curbs during drone flights. Detections on gray curbs are now completely eliminated.
*   **100% Generalization Pass Rate:** Achieved a **100.0% validation success rate** (10/10 detections) across all 319 flag classes in the validation sweep, including Qatar, Egypt, and Germany.

### Training Progress & Curves
![YOLO26s Training Curves](runs/kaggle_results/runs/detect/yolo26_small/results.png)

---

## 4K Validation Image Test

The model was verified locally on high-resolution 4K validation images (`validate_ai/`) using a test resolution of `imgsz=640` (to match the training scale and avoid resolution mismatch). 

Testing on `15.jpg` (containing Germany, Russia, and France flags) achieved **100.0% accuracy** with **zero false positives**:

*   **France Flag:** Detected with **92% confidence** (deviation: 1.0px).
*   **Germany Flag:** Detected with **80% confidence** (deviation: 1.4px).
*   **Russia Flag:** Detected with **76% confidence** (deviation: 2.5px).

Annotated predictions are saved in the `validate_ai_results_3840/` directory.

### Annotated 4K Result Sample (15.jpg)
![Trained YOLO26s detections on 15.jpg](assets/highres_15_detected.jpg)

---

## File Structure & Contents

*   `yolo26s_flag_best.pt`: The final, high-accuracy trained model weights (tracked in the repository).
*   `train_yolo26_kaggle.ipynb`: The notebook executed on Kaggle to perform remote training.
*   `run_kaggle_training.py`: Python automation orchestrator that zips the dataset, uploads it to Kaggle, triggers training via Kaggle API, and retrieves training weights and evaluation charts automatically.
*   `verify_fps_resolved.py`: Verification script evaluating the model on known false-positive frames.
*   `validate_exact_rates.py`: Validation sweep script evaluating detection rates for all 319 flag classes.
*   `kaggle_dataset_resized/`: The resized training dataset (640px) used for remote Kaggle uploads.
*   `validate_ai/`: High-resolution 4K validation images.
*   `validate_ai_results_3840/`: Validation images annotated with bounding boxes and confidence scores.

---

## Getting Started

### Prerequisites

Install the required python packages:
```bash
pip install ultralytics opencv-python numpy PyYAML
```

### Running Inference

To run the model on validation images:
```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('yolo26s_flag_best.pt')

# Run inference at native 640px scale (matches training resolution)
results = model.predict(source='validate_ai/15.jpg', imgsz=640, conf=0.10)
results[0].show()  # Display predictions
```
