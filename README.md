# Traffic Accident Detection with YOLOv8

A Python-based traffic monitoring system that captures live camera feeds from TDOT (Tennessee Department of Transportation) traffic cameras and uses YOLOv8 object detection to identify vehicles and analyze traffic patterns for potential accident detection.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [How It Works](#how-it-works)
- [Usage](#usage)
- [Configuration](#configuration)
- [Output](#output)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)
- [Technical Details](#technical-details)

## 🎯 Overview

This project monitors live traffic camera feeds and uses computer vision (YOLOv8) to detect vehicles in real-time. The system captures screenshots from traffic cameras, runs object detection to identify vehicles, and analyzes the data for potential accident indicators such as overlapping vehicles or unusual traffic patterns.

## ✨ Features

- **Automated Screenshot Capture**: Uses Selenium to access live traffic camera feeds
- **Real-time Object Detection**: YOLOv8 identifies vehicles (cars, trucks, buses, motorcycles) and other objects
- **Visual Annotations**: Saves images with bounding boxes around detected objects
- **Traffic Analysis**: Analyzes vehicle positions, counts, and potential collisions
- **Batch Processing**: Processes multiple frames and generates aggregate statistics
- **Overlap Detection**: Identifies overlapping vehicles as potential collision indicators

## 📁 Project Structure

```
Traffic-Accident-Detection/
├── yolo_screenshot_detector.py    # Main capture & detection script
├── detection_analysis_example.py  # Analysis script for captured images
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── implementation_plan.md         # Future development roadmap
├── open-url-sample.py            # Original simple screenshot script
│
├── captures/                      # Raw screenshots (auto-created)
│   └── YYYY-MM-DD-HH-MM-SS.png
│
├── captures_annotated/            # Annotated images with bounding boxes (auto-created)
│   └── YYYY-MM-DD-HH-MM-SS.png
│
└── yolov8n.pt                    # YOLOv8 nano model (auto-downloaded)
```

## 🔧 Prerequisites

Before you begin, ensure you have the following installed:

### 1. Python
- **Version**: Python 3.8 or higher
- **Download**: https://www.python.org/downloads/
- **Verify installation**: 
  ```bash
  python --version
  ```

### 2. Web Browser
- **Firefox** (recommended) - Script is configured for Firefox
- Alternative: Chrome/Edge (requires script modification)

### 3. Browser Driver
- **geckodriver** for Firefox
  - **Download**: https://github.com/mozilla/geckodriver/releases
  - **Installation**: 
    - Extract the executable
    - Add to your system PATH, OR
    - Place in your Python Scripts directory

**Verify geckodriver installation:**
```bash
geckodriver --version
```

## 🚀 Installation

### Step 1: Clone or Download the Repository

```bash
cd C:\git
git clone <repository-url> Traffic-Accident-Detection
cd Traffic-Accident-Detection
```

Or download and extract the ZIP file to `C:\git\Traffic-Accident-Detection`

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- **ultralytics** - YOLOv8 framework
- **opencv-python** - Image processing
- **numpy** - Numerical operations
- **torch & torchvision** - Deep learning framework
- **selenium** - Browser automation
- **pillow** - Image handling

**Note**: First run will automatically download the YOLOv8n model (~6MB).

### Step 3: Verify Installation

Test that all packages are installed correctly:

```bash
python -c "from ultralytics import YOLO; import cv2; from selenium import webdriver; print('✓ All packages installed successfully!')"
```

## 🔍 How It Works

### Script 1: `yolo_screenshot_detector.py`

This is the main capture and detection script. Here's what it does:

#### **Step-by-Step Process:**

1. **Initialization**
   - Loads the YOLOv8 nano model (`yolov8n.pt`)
   - Clears previous captures from `captures/` and `captures_annotated/` folders
   - Creates fresh directories

2. **Browser Setup**
   - Opens Firefox browser using Selenium
   - Maximizes the window for full view
   - Navigates to TDOT traffic camera URL
   - Waits 5 seconds for page to fully load

3. **Capture Loop** (runs 5 times by default, every 2 seconds)
   - Takes a screenshot using Selenium's built-in capture (browser window only)
   - Converts PNG bytes to OpenCV image format
   - Saves raw screenshot to `captures/` folder

4. **YOLO Detection**
   - Runs YOLOv8 object detection on the captured image
   - Identifies objects (cars, trucks, buses, motorcycles, people, etc.)
   - Records bounding boxes, class labels, and confidence scores

5. **Annotation & Saving**
   - Draws bounding boxes with labels on the image
   - Saves annotated image to `captures_annotated/` folder
   - Prints detection details to console

6. **Cleanup**
   - Closes the browser
   - Displays summary of saved files

#### **Key Code Sections:**

```python
# Browser screenshot (not full screen)
png_bytes = browser.get_screenshot_as_png()
image = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)

# YOLO inference
results = model(image)
result = results[0]

# Draw annotations
annotated_frame = result.plot()
```

### Script 2: `detection_analysis_example.py`

This script performs detailed analysis on captured images.

#### **Step-by-Step Process:**

1. **Image Discovery**
   - Scans `captures_annotated/` folder for all PNG files
   - Sorts images by filename (timestamp order)

2. **Per-Image Analysis**
   - Loads each annotated image
   - Re-runs YOLO detection to extract data
   - Identifies vehicle types and positions
   - Calculates bounding box centers and areas

3. **Overlap Detection**
   - Compares all vehicle bounding boxes
   - Calculates IoU (Intersection over Union)
   - Flags overlaps > 20% as potential collisions
   - Counts high-confidence detections

4. **Aggregate Statistics**
   - Total vehicles across all frames
   - Average vehicles per frame
   - Frame with most vehicles
   - Frames with overlaps (⚠️ potential accidents)

5. **Future Recommendations**
   - Provides notes on implementing advanced accident detection
   - Suggests frame-to-frame tracking
   - Recommends custom model training

#### **Key Analysis Features:**

```python
# IoU calculation for overlap detection
def boxes_overlap(box1, box2, threshold=0.2):
    # Calculates intersection over union
    # Returns True if IoU > threshold
```

## 🎮 Usage

### Quick Start

1. **Run the capture script**:
   ```bash
   python yolo_screenshot_detector.py
   ```
   
   **What happens:**
   - Browser opens automatically
   - Captures 5 screenshots over ~10 seconds
   - Displays detections in console
   - Saves images to folders
   - Browser closes automatically

2. **Analyze the captured images**:
   ```bash
   python detection_analysis_example.py
   ```
   
   **What happens:**
   - Processes all images in `captures_annotated/`
   - Displays detailed analysis for each image
   - Shows aggregate statistics
   - Identifies potential accident indicators

### Expected Console Output

#### From `yolo_screenshot_detector.py`:

```
Clearing old images...
✓ Folders ready for new captures

Saved screenshot: captures\2025-11-13-0-8-37.png
Saved annotated image: captures_annotated\2025-11-13-0-8-37.png

Detections in frame 1:
  Class: car (ID: 2), Confidence: 0.89, Box: (245.3, 156.7, 389.2, 234.5)
  Class: truck (ID: 7), Confidence: 0.76, Box: (512.1, 178.3, 687.9, 298.4)
  Class: car (ID: 2), Confidence: 0.82, Box: (123.4, 201.2, 245.6, 289.1)

...

✓ Processing complete!
✓ Raw screenshots saved in: captures/
✓ Annotated images saved in: captures_annotated/
```

#### From `detection_analysis_example.py`:

```
Found 5 images to analyze

================================================================================
ANALYZING IMAGE 1/5: 2025-11-13-0-8-37.png
================================================================================

Total detections: 12
  Detection 1: car (conf: 0.89)
  Detection 2: truck (conf: 0.76)
  Detection 3: car (conf: 0.82)
  ...

--- Analysis ---
Vehicle count: 8
  ✓ No significant overlaps detected
  High-confidence vehicles (>0.7): 6

...

================================================================================
SUMMARY ACROSS ALL FRAMES
================================================================================

Total images analyzed: 5
Total vehicles detected: 37
Total overlaps detected: 0
Average vehicles per frame: 7.4

Frame with most vehicles: 2025-11-13-0-8-42.png (9 vehicles)

✓ No overlaps detected in any frame
```

## ⚙️ Configuration

### Change Camera URL

Edit line 29 in `yolo_screenshot_detector.py`:

```python
url = "https://smartway.tn.gov/allcams/camera/3245"  # Change camera ID here
```

**Finding other cameras:**
- Visit: https://smartway.tn.gov/traffic
- Browse available cameras
- Copy the camera URL

### Change Number of Captures

Edit line 33 in `yolo_screenshot_detector.py`:

```python
for i in range(5):  # Change 5 to desired number of screenshots
```

### Change Capture Interval

Edit line 34 in `yolo_screenshot_detector.py`:

```python
time.sleep(2)  # Change 2 to desired seconds between captures
```

### Use Different YOLO Model

Edit line 13 in `yolo_screenshot_detector.py`:

```python
model = YOLO("yolov8n.pt")  # Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
```

**Model Comparison:**

| Model    | Size  | Speed      | Accuracy | Use Case              |
|----------|-------|------------|----------|-----------------------|
| yolov8n  | ~6MB  | Fastest    | Good     | Real-time, testing    |
| yolov8s  | ~22MB | Fast       | Better   | Balanced              |
| yolov8m  | ~50MB | Moderate   | High     | Accuracy priority     |
| yolov8l  | ~87MB | Slow       | Higher   | High-end systems      |
| yolov8x  | ~131MB| Slowest    | Highest  | Maximum accuracy      |

### Adjust Overlap Detection Threshold

Edit line 88 in `detection_analysis_example.py`:

```python
def boxes_overlap(box1, box2, threshold=0.2):  # 0.2 = 20% IoU threshold
```

Lower threshold = more sensitive to overlaps  
Higher threshold = only detects significant overlaps

## 📊 Output

### Folder Structure After Running

```
captures/
├── 2025-11-13-0-8-34.png  # Raw screenshot
├── 2025-11-13-0-8-37.png
├── 2025-11-13-0-8-39.png
├── 2025-11-13-0-8-42.png
└── 2025-11-13-0-8-44.png

captures_annotated/
├── 2025-11-13-0-8-34.png  # Same image with bounding boxes
├── 2025-11-13-0-8-37.png
├── 2025-11-13-0-8-39.png
├── 2025-11-13-0-8-42.png
└── 2025-11-13-0-8-44.png
```

### What's in the Annotated Images?

- **Bounding boxes** around detected objects
- **Class labels** (car, truck, bus, etc.)
- **Confidence scores** (0-1, shown as percentage)
- **Color-coded boxes** by object class

### Detection Classes

YOLOv8 (COCO dataset) can detect 80 classes. Relevant for traffic:

| Class ID | Name          | Description           |
|----------|---------------|-----------------------|
| 0        | person        | Pedestrians           |
| 2        | car           | Standard vehicles     |
| 3        | motorcycle    | Motorcycles           |
| 5        | bus           | Buses                 |
| 7        | truck         | Trucks                |
| 9        | traffic light | Traffic signals       |
| 11       | stop sign     | Stop signs            |

## 🐛 Troubleshooting

### Error: "geckodriver not found"

**Problem**: Selenium can't find the browser driver.

**Solution**:
1. Download geckodriver: https://github.com/mozilla/geckodriver/releases
2. Extract the executable
3. Add to PATH:
   - Windows: Add folder to System Environment Variables
   - Or place `geckodriver.exe` in Python's Scripts folder

### Error: "No module named 'ultralytics'"

**Problem**: Dependencies not installed.

**Solution**:
```bash
pip install -r requirements.txt
```

### Error: "CUDA not available" or Slow Performance

**Problem**: Running on CPU instead of GPU.

**Solution** (Optional - for GPU acceleration):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Note**: CPU is sufficient for this project. GPU provides faster processing.

### Browser Opens but Shows Blank Screenshots

**Problem**: Page not fully loaded before screenshot.

**Solution**: Increase wait time (line 31 in `yolo_screenshot_detector.py`):
```python
time.sleep(10)  # Increase from 5 to 10 seconds
```

### Error: "No images found in captures_annotated/ folder"

**Problem**: Haven't run the capture script yet.

**Solution**: Run capture script first:
```bash
python yolo_screenshot_detector.py
```

### Firefox Not Opening

**Problem**: Browser path not found or using wrong browser.

**Solution 1** - Use Chrome instead:
```python
browser = webdriver.Chrome()  # Replace line 27
```

**Solution 2** - Specify Firefox path:
```python
from selenium.webdriver.firefox.service import Service
service = Service('C:\\Path\\To\\Firefox\\firefox.exe')
browser = webdriver.Firefox(service=service)
```

### Script Runs But No Detections

**Problem**: Camera view may be empty or quality issue.

**Check**:
- Verify camera URL is working in regular browser
- Check if it's nighttime (lower detection accuracy)
- Try different camera angle/location
- Lower confidence threshold if needed

## 🚦 Tracking & Accident Detection (pipeline.py)

The new `pipeline.py` adds **DeepSORT-based vehicle tracking** and a continuous
**accident confidence score** on top of the existing YOLO detector.

### How the Pipeline Works

```
HLS stream / video / images
        │
        ▼
  YOLO detection          ← vehicle classes only (car, bus, truck, motorcycle)
        │  (x1,y1,x2,y2, conf, class)
        ▼
  DeepSORT tracker        ← assigns persistent track IDs across frames
        │  confirmed Track objects
        ▼
  AccidentScorer          ← rolling 5-second window of per-track history
        │  score ∈ [0,1], accident_detected bool, metadata
        ▼
  Annotated output frame  ← bounding boxes + "ID N class" labels + score banner
```

### Installation

Install the additional dependency:

```bash
pip install deep-sort-realtime>=1.3.2
# or just re-run:
pip install -r requirements.txt
```

> `deep-sort-realtime` ships a built-in MobileNet Re-ID extractor — no extra
> model download is required.

### Running the Pipeline

```bash
# Default: live TDOT HLS stream
python pipeline.py

# Local video file
python pipeline.py --source path/to/video.mp4

# Folder of images (processed in filename order)
python pipeline.py --source frames/

# Custom HLS / RTSP URL
python pipeline.py --source https://example.com/stream.m3u8

# Test mode — processes 300 frames, prints per-frame scores, then exits
python pipeline.py --source video.mp4 --test

# Headless (no preview window — for servers)
python pipeline.py --source video.mp4 --no-display

# Combine flags
python pipeline.py --source video.mp4 --test --no-display --max-test-frames 100
```

### Console Output

```
[00042] score=0.031  stop=0.00  decel=0.00  iou=0.00  post=0.00  anomaly=0.00  tracks=7
[00043] score=0.044  stop=0.00  decel=0.12  iou=0.00  post=0.00  anomaly=0.00  tracks=7
[00081] score=0.712 *** ACCIDENT DETECTED ***  stop=0.50  decel=0.45  iou=0.80  post=0.80  anomaly=0.50  tracks=5
         involved tracks: [3, 9]  region: (412.0, 210.0, 587.0, 334.0)
```

Annotated frames are saved to `pipeline_output/` (JPEG by default).

### Accident Confidence Signals

Five heuristic signals are computed from each track's rolling history window
(last 5 seconds) and combined into a weighted score:

| Signal | Weight | Description |
|---|---|---|
| `sudden_stop` | 0.25 | A track's speed drops below threshold and stays near-zero |
| `abrupt_decel` | 0.20 | High negative mean acceleration over a short window |
| `collision_iou` | 0.30 | Two tracks' boxes overlap **and** were converging beforehand |
| `post_collision` | 0.15 | After overlap: track stationary or moving erratically |
| `traffic_anomaly` | 0.10 | Multiple nearby vehicles also slow near the event region |

**Scoring pipeline:**
```
raw_score      = Σ (weight × signal)
smoothed_score = EMA(raw_score, α=0.30)

accident_detected = True   when smoothed_score ≥ 0.65
accident_detected = False  when smoothed_score < 0.40  (hysteresis)
```

### Configuration

All parameters live in `config.py` — edit once and every module picks up the change:

| Parameter | Default | Description |
|---|---|---|
| `VEHICLE_CLASSES` | `[2,3,5,7]` | COCO IDs: car, motorcycle, bus, truck |
| `DET_CONF_THRESHOLD` | `0.30` | Minimum YOLO detection confidence |
| `HISTORY_WINDOW_SECONDS` | `5` | Rolling history length per track |
| `DEEPSORT_MAX_AGE` | `30` | Frames before a lost track is deleted |
| `STOP_SPEED_THRESHOLD` | `5.0 px/frame` | Speed below which a vehicle is "stopped" |
| `COLLISION_IOU_THRESHOLD` | `0.10` | Min box IoU to flag a collision event |
| `EMA_ALPHA` | `0.30` | Score smoothing strength (higher = faster) |
| `HIGH_THRESHOLD` | `0.65` | Score level that sets `accident_detected = True` |
| `LOW_THRESHOLD` | `0.40` | Score level that clears `accident_detected` |
| `OUTPUT_DIR` | `pipeline_output` | Directory for saved annotated frames |

### New Files

| File | Purpose |
|---|---|
| `pipeline.py` | Main entry point — CLI with `--source`, `--test`, `--no-display` |
| `tracker.py` | `DeepSortTracker` — wraps DeepSort, handles xyxy ↔ xywh conversion |
| `accident_scorer.py` | `AccidentScorer` — rolling history, five signals, EMA + hysteresis |
| `config.py` | Central configuration for all modules |

### Coordinate Format

- **YOLO output**: `xyxy` (x1, y1, x2, y2) pixel coords
- **DeepSORT input**: `xywh` (left, top, width, height) — converted inside `tracker.py`
- **Internal history**: `cx, cy` (box centre) used for velocity and acceleration maths

---

## 🚀 Future Enhancements

### Planned Features (See `implementation_plan.md`)

1. **Frame-to-Frame Tracking**
   - Track individual vehicles across frames
   - Detect sudden stops or unusual movements
   - Identify stationary vehicles in traffic lanes

2. **Anomaly Detection**
   - Establish baseline "normal traffic" patterns
   - Flag unusual vehicle clustering
   - Detect debris or people on roadway

3. **Custom Model Training**
   - Train YOLOv8 on accident-specific dataset
   - Binary classification: accident vs. normal
   - Use datasets like CADP or DADA-2000

4. **Multi-Camera Support**
   - Monitor multiple cameras simultaneously
   - Parallel processing with threading
   - Unified dashboard for all feeds

5. **Real-Time Streaming**
   - Replace screenshot approach with video stream
   - Direct MJPEG/HLS stream capture
   - Continuous monitoring (24/7)

6. **Alert System**
   - Email/SMS notifications on accident detection
   - Desktop notifications
   - Log file with timestamps and confidence scores

7. **Web Dashboard**
   - Live view of all cameras
   - Historical incident log
   - Statistics and analytics

## 🔬 Technical Details

### YOLOv8 Architecture

- **Framework**: Ultralytics YOLOv8
- **Model**: Nano (yolov8n.pt) - 3.2M parameters
- **Input**: 640x640 RGB images (auto-resized)
- **Output**: Bounding boxes, class labels, confidence scores
- **Speed**: ~20-50ms per image (CPU), ~5-10ms (GPU)

### Object Detection Pipeline

```
Input Image → Preprocessing → YOLOv8 CNN → Post-processing → Detections
             (resize, norm)                  (NMS, filtering)   (boxes, labels)
```

### Overlap Detection Algorithm

Uses **Intersection over Union (IoU)**:

```
IoU = Area of Overlap / Area of Union

If IoU > threshold (0.2):
    Flag as potential collision
```

### Performance Metrics

- **Capture Rate**: ~1 frame per 2 seconds
- **Processing Time**: ~100-500ms per frame (CPU)
- **Detection Accuracy**: ~80-95% for vehicles (daylight, clear conditions)
- **Memory Usage**: ~200-500MB

### Dependencies Explained

- **ultralytics**: Modern YOLOv8 implementation
- **opencv-python**: Image processing, encoding/decoding
- **numpy**: Array operations for image data
- **torch**: PyTorch deep learning framework (YOLOv8 backend)
- **selenium**: Browser automation for camera access
- **pillow**: Additional image format support

## 📝 License

This project is for educational and research purposes. 

**Important**:
- Respect TDOT camera usage policies
- Do not use for commercial purposes without permission
- Traffic camera feeds are public but should be used responsibly

## 🤝 Contributing

To contribute:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Support

For issues or questions:
- Check the [Troubleshooting](#troubleshooting) section
- Review YOLOv8 documentation: https://docs.ultralytics.com/
- Selenium documentation: https://selenium-python.readthedocs.io/

## 🎓 Learning Resources

- **YOLOv8 Tutorial**: https://docs.ultralytics.com/modes/predict/
- **Computer Vision Basics**: https://opencv.org/
- **Selenium WebDriver**: https://www.selenium.dev/documentation/
- **COCO Dataset Classes**: https://cocodataset.org/#explore

---

**Project Status**: Active Development  
**Last Updated**: November 2025  
**Python Version**: 3.8+  
**Platform**: Windows (with minor changes works on Linux/Mac)

---

Made with ❤️ for traffic safety and computer vision learning
