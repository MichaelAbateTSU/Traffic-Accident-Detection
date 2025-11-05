# Traffic Accident Detection AI - Implementation Plan

## Overview

Build an end-to-end traffic accident detection system using PyTorch with YOLOv8, leveraging pretrained models and public accident datasets for transfer learning. The system will process live TDOT camera feeds and detect accidents in real-time with console logging and file alerts.

## Project Structure

Create organized folder structure:

```
/data
  /raw              # Downloaded dataset frames
  /processed        # Preprocessed images
  /annotations      # YOLO format labels
/models
  /pretrained       # Downloaded base models
  /trained          # Fine-tuned models
/src
  data_loader.py    # Stream capture and preprocessing
  model.py          # YOLOv8 model wrapper
  detector.py       # Real-time detection logic
  utils.py          # Helper functions
/alerts             # Saved accident frames/clips
/logs               # Event logs
/notebooks          # Training experiments (optional)
main.py             # Main detection script
train.py            # Model training script
config.py           # Configuration settings
requirements.txt
README.md
```

## Implementation Steps

### 1. Environment Setup and Dependencies

**File: `requirements.txt`**

- Core: `opencv-python`, `numpy`, `torch`, `torchvision`, `ultralytics` (YOLOv8)
- Data: `requests`, `pillow`, `albumentations`
- Utilities: `python-dotenv`, `pyyaml`, `tqdm`
- Optional: `scikit-learn`, `matplotlib`, `pandas`

**File: `config.py`**

- Camera URLs (TDOT feeds)
- Model paths and hyperparameters
- Detection thresholds and confidence scores
- Alert settings (save paths, notification preferences)
- GPU/CPU device selection

### 2. Data Collection and Preprocessing

**File: `src/data_loader.py`**

Replace Selenium-based screenshot approach with direct stream capture:

- Implement `StreamCapture` class using `cv2.VideoCapture()` for TDOT camera URLs
- Handle MJPEG/HLS stream parsing without browser overhead
- Frame extraction at configurable intervals (every 2-3 seconds or motion-based)
- Preprocessing pipeline: resize to 640x640, normalization, format conversion

Key functions:

- `capture_stream(url)` - Connect to live camera feed
- `extract_frames()` - Continuous frame grabbing with error handling
- `preprocess_frame(frame)` - Resize, normalize for YOLO input
- `save_frame(frame, label, timestamp)` - Organize into folders

**Download public dataset:**

- Use CADP (Crash Accident Detection and Prediction) or DADA-2000 dataset
- Script to download and organize into YOLO format (images + txt annotations)
- Split into train/val/test (70/15/15)

### 3. YOLOv8 Model Setup and Training

**File: `src/model.py`**

Implement YOLOv8-based accident detection:

- Load pretrained YOLOv8 (`yolov8n.pt` or `yolov8s.pt`) from Ultralytics
- Configure for binary classification: "normal" vs "accident"
- Option 1: Fine-tune on accident dataset with frozen backbone
- Option 2: Use pretrained vehicle detection + anomaly scoring based on motion/trajectory

Key components:

- `AccidentDetector` class wrapping YOLO model
- `load_model(weights_path)` - Initialize with pretrained or custom weights
- `predict(frame)` - Run inference and return detections with confidence
- GPU acceleration check (`torch.cuda.is_available()`)

**File: `train.py`**

Training script for fine-tuning:

- Load YOLO model with pretrained COCO weights
- Configure data.yaml pointing to accident dataset
- Training loop with hyperparameters: epochs=50, batch_size=16, img_size=640
- Validation during training
- Save best model to `/models/trained/accident_detector.pt`
- Output metrics: precision, recall, mAP, F1-score
- Generate confusion matrix and loss curves

### 4. Real-Time Detection Pipeline

**File: `main.py`**

Main execution script:

```python
# Pseudocode structure:
1. Load trained YOLOv8 model
2. Initialize StreamCapture for TDOT camera
3. Continuous loop:
   - Capture frame
   - Preprocess
   - Run detection
   - If accident detected (confidence > threshold):
     * Log event with timestamp, camera ID, confidence
     * Save frame to /alerts folder
     * Print console notification
     * Optional: send email/desktop notification
   - Handle disconnections and errors
4. Cleanup on exit
```

**File: `src/detector.py`**

Core detection logic:

- `RealTimeDetector` class managing the pipeline
- `process_frame(frame)` - Complete detection workflow
- `is_accident(detections)` - Decision logic based on confidence threshold
- `log_event(event_data)` - Write to log file
- `save_alert(frame, metadata)` - Save with timestamp naming
- Multi-camera support via threading/multiprocessing

### 5. Logging and Alert System

**File: `src/utils.py`**

Utilities for logging and notifications:

- `setup_logger()` - Configure file and console logging
- `send_notification(message)` - Desktop notification using `plyer` or console print
- `save_clip(frames, path)` - Save short video clip (last 5-10 frames)
- `generate_report()` - Summary of detected events
- Error handling for stream failures and reconnection logic

**Log format:**

```
[2025-11-05 14:23:45] ACCIDENT DETECTED | Camera: 3245 | Confidence: 0.87 | Frame: alerts/2025-11-5-14-23-45.jpg
```

### 6. Performance Optimization

- Inference optimization: FP16 precision if GPU available
- Frame skipping during processing to maintain real-time performance
- Batch processing for multiple cameras
- Asyncio for non-blocking stream capture
- Configurable detection interval (process every Nth frame)

### 7. Documentation

**File: `README.md`**

Include:

- Project overview and objectives
- Architecture diagram (simple text-based)
- Setup instructions: environment creation, dependency installation
- Dataset download and preparation steps
- Training instructions: `python train.py --epochs 50`
- Running detection: `python main.py --camera 3245`
- Configuration options in config.py
- Example outputs and performance metrics
- Troubleshooting common issues (stream connection, GPU setup)

## Key Technical Decisions

1. **YOLOv8 from Ultralytics**: Modern, well-documented, easy transfer learning
2. **Direct stream capture**: Remove Selenium/PyAutoGUI overhead for efficiency
3. **Transfer learning**: Use CADP dataset + COCO pretrained weights (faster than training from scratch)
4. **Single-stage detection**: YOLOv8 detects vehicles + abnormal patterns in one pass
5. **Binary classification**: Simplify to accident/normal (can expand later)
6. **Console-based alerts**: Lightweight, production-ready logging without web overhead

## Expected Deliverables

- ✅ Trained YOLOv8 model (`accident_detector.pt`)
- ✅ Real-time detection script processing live TDOT feeds
- ✅ Automatic accident logging with saved frames in `/alerts`
- ✅ Performance metrics from training (precision, recall, F1)
- ✅ Complete documentation for setup and execution
- ✅ Clean, modular codebase ready for extension

## Testing and Validation

- Test on held-out dataset: confusion matrix, precision, recall, F1
- Live testing on multiple TDOT camera feeds
- Measure inference time (target: <100ms per frame for real-time)
- Stress test: handle disconnections, reconnections gracefully
- Document false positive/negative rates