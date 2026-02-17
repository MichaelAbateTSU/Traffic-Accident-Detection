import time
import datetime
import numpy as np

from ultralytics import YOLO
import cv2
import os
import shutil

# -------- STREAM CONFIGURATION --------
STREAM_URL       = "https://mcleansfs3.us-east-1.skyvdn.com/rtplive/R2_066/playlist.m3u8"
CAPTURE_INTERVAL = 1   # Seconds between each captured frame
CAPTURE_COUNT    = 5   # Total number of frames to grab

# -------- PERFORMANCE SETTINGS --------
SAVE_IMAGES  = True   # Set to False to benchmark pure detection speed
USE_JPEG     = True   # JPEG is 5-10x faster to save than PNG
JPEG_QUALITY = 85     # 0-100, higher = better quality but slower

# -------- DETECTION CONFIGURATION --------
CONF_THRESHOLD  = 0.15   # Lower threshold for distant vehicles
IOU_THRESHOLD   = 0.45   # Intersection over Union for NMS
INFERENCE_SIZE  = 1280   # Larger size for better small-object detection

# Traffic-relevant COCO classes
TRAFFIC_CLASSES = [0, 1, 2, 3, 5, 7, 9, 11]  # person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign
FILTER_CLASSES  = True   # Set to False to show all 80 COCO classes for debugging

# Debug settings
DEBUG_FRAMES = True   # Print frame quality diagnostics
DEBUG_TIMING = True   # Print inference timing

# -------- YOLO SETUP --------
# yolo11x.pt is the largest/most accurate model in the Ultralytics YOLO11 family.
# It will be downloaded automatically (~110 MB) if not already present.
model = YOLO("yolo11x.pt")

print(f"Model: {model.model_name if hasattr(model, 'model_name') else 'YOLO11x'}")
print(f"Inference size: {INFERENCE_SIZE}x{INFERENCE_SIZE}")
print(f"Confidence threshold: {CONF_THRESHOLD}")
print(f"Class filtering: {'ON (traffic only)' if FILTER_CLASSES else 'OFF (all classes)'}\n")

# -------- FOLDERS --------
print("Clearing old images...")
if os.path.exists("captures"):
    shutil.rmtree("captures")
if os.path.exists("captures_annotated"):
    shutil.rmtree("captures_annotated")

os.makedirs("captures")
os.makedirs("captures_annotated")
print("✓ Folders ready\n")

# -------- PHASE 1: CAPTURE FRAMES FROM STREAM --------
print(f"Connecting to stream: {STREAM_URL}")
cap = cv2.VideoCapture(STREAM_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer to get the most recent frame

if not cap.isOpened():
    raise RuntimeError(
        "Could not open stream. Make sure FFmpeg is available to OpenCV.\n"
        "You may need to reinstall: pip install opencv-python --upgrade"
    )

print(f"✓ Stream connected. Grabbing {CAPTURE_COUNT} frame(s), one every {CAPTURE_INTERVAL}s ...\n")

captured_frames = []

for i in range(CAPTURE_COUNT):
    ok, frame = cap.read()

    if not ok or frame is None:
        raise RuntimeError(
            f"Failed to read frame {i + 1} from stream. "
            "The stream may have dropped or returned an empty frame."
        )

    t = datetime.datetime.now()
    base_name = f"{t.year}-{t.month:02d}-{t.day:02d}-{t.hour:02d}-{t.minute:02d}-{t.second:02d}"

    # Save raw frame
    if USE_JPEG:
        filepath = os.path.join("captures", f"{base_name}.jpg")
        cv2.imwrite(filepath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    else:
        filepath = os.path.join("captures", f"{base_name}.png")
        cv2.imwrite(filepath, frame)

    print(f"[{i + 1}/{CAPTURE_COUNT}] Saved frame: {filepath}")

    if DEBUG_FRAMES:
        print(f"  Shape={frame.shape}, dtype={frame.dtype}, "
              f"range=[{frame.min()}, {frame.max()}]")

    captured_frames.append((filepath, frame.copy()))

    # Wait before next capture (skip wait after last one)
    if i < CAPTURE_COUNT - 1:
        time.sleep(CAPTURE_INTERVAL)

cap.release()
print(f"\n✓ All {CAPTURE_COUNT} frames captured.\n")

# -------- PHASE 2: RUN YOLO ON EACH FRAME --------
print("Running YOLO detection on captured frames...\n")

for idx, (filepath, frame) in enumerate(captured_frames):
    print(f"--- Frame {idx + 1}/{CAPTURE_COUNT}: {os.path.basename(filepath)} ---")

    if DEBUG_TIMING:
        t0 = time.time()

    results = model.predict(
        frame,
        imgsz=INFERENCE_SIZE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        classes=TRAFFIC_CLASSES if FILTER_CLASSES else None,
        verbose=False
    )

    result = results[0]

    if DEBUG_TIMING:
        inference_ms = (time.time() - t0) * 1000
        print(f"  Inference time: {inference_ms:.1f}ms ({1000 / inference_ms:.1f} FPS max)")

    # Save annotated image
    try:
        annotated_frame = result.plot(
            line_width=3,
            font_size=1.5,
            labels=True,
            conf=True,
            boxes=True
        )
    except TypeError:
        annotated_frame = result.plot()
        print("  (using default plot parameters)")

    base_name = os.path.basename(filepath)
    annotated_path = os.path.join("captures_annotated", base_name)

    if SAVE_IMAGES:
        if USE_JPEG:
            cv2.imwrite(annotated_path, annotated_frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        else:
            cv2.imwrite(annotated_path, annotated_frame)
        print(f"  Saved annotated: {annotated_path}")

    # Print detections
    boxes = result.boxes
    total_detections = len(boxes)
    high_conf_count = sum(1 for box in boxes if float(box.conf[0]) >= 0.5)

    print(f"  Detections: {total_detections} total, {high_conf_count} high-confidence")

    if FILTER_CLASSES and total_detections == 0:
        print("  ℹ No traffic objects detected. Try setting FILTER_CLASSES=False to debug.")
    elif not FILTER_CLASSES and total_detections == 0:
        print("  ℹ No detections at all - check frame quality or inference size.")

    for box in boxes:
        cls        = int(box.cls[0])
        conf       = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_name = result.names[cls]
        marker     = "★" if conf >= 0.5 else " "
        print(f"  {marker} {class_name}: {conf:.2%} at ({x1:.0f}, {y1:.0f})")

    print()

# -------- DONE --------
print("✓ Processing complete!")
print(f"✓ Raw screenshots in:      captures/")
print(f"✓ Annotated images in:     captures_annotated/")
