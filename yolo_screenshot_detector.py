import time
import datetime
import numpy as np

from ultralytics import YOLO
import cv2
import os
import shutil

# -------- STREAM CONFIGURATION --------
STREAM_URL = "https://mcleansfs5.us-east-1.skyvdn.com/rtplive/R2_133/playlist.m3u8"
TARGET_FPS = 5  # Process 5 frames per second (up from 1 for better tracking)
FRAME_COUNT = 50  # More frames to see tracking in action

# -------- PERFORMANCE SETTINGS --------
SAVE_IMAGES = True  # Set to False to benchmark pure detection speed
USE_JPEG = True  # JPEG is 5-10x faster to save than PNG
JPEG_QUALITY = 85  # 0-100, higher = better quality but slower

# -------- DETECTION CONFIGURATION --------
CONF_THRESHOLD = 0.15  # Lower threshold for distant vehicles
IOU_THRESHOLD = 0.45   # Intersection over Union for NMS
INFERENCE_SIZE = 1280  # Larger size crucial for highway cams (default: 640)

# Traffic-relevant COCO classes
TRAFFIC_CLASSES = [0, 1, 2, 3, 5, 7, 9, 11]  # person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign
FILTER_CLASSES = True  # Set to False to show all 80 COCO classes for debugging

# Tracking settings
ENABLE_TRACKING = True  # Use ByteTrack for stable vehicle IDs across frames

# Debug settings
DEBUG_FRAMES = True  # Print frame quality diagnostics
DEBUG_TIMING = True  # Print inference timing

# -------- YOLO SETUP --------
model = YOLO("yolo26x.pt")  # YOLO26 Extra-Large - latest flagship (Jan 2026)

# Verify model loaded
print(f"Model: {model.model_name if hasattr(model, 'model_name') else 'YOLO26x'}")
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
print("✓ Folders ready for new captures\n")

# -------- STREAM SETUP --------
print(f"Connecting to stream: {STREAM_URL}")
cap = cv2.VideoCapture(STREAM_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering for lower latency

if not cap.isOpened():
    raise RuntimeError(
        "Could not open stream. Make sure FFmpeg is available to OpenCV.\n"
        "You may need to reinstall: pip install opencv-python --upgrade"
    )

print("✓ Stream connected successfully")

# Detect stream properties
stream_fps = cap.get(cv2.CAP_PROP_FPS)
stream_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
stream_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Stream properties: {stream_width}x{stream_height}")
if stream_fps > 0:
    print(f"Stream FPS: {stream_fps:.1f}")
else:
    print("Stream FPS: unknown (HLS)")

if stream_height < 480:
    print(f"⚠ Low resolution stream ({stream_width}x{stream_height}) - distant objects may be hard to detect")
print()

# -------- STREAM CAPTURE LOOP --------
frame_interval = 1 / TARGET_FPS
last_capture_time = 0
frames_processed = 0
reconnect_attempts = 0
MAX_RECONNECT_ATTEMPTS = 3

print(f"Processing {FRAME_COUNT} frames at {TARGET_FPS} FPS...\n")

while frames_processed < FRAME_COUNT:
    ok, frame = cap.read()
    
    if not ok:
        # Stream hiccup - attempt reconnection
        print(f"⚠ Stream read failed. Attempting reconnection ({reconnect_attempts + 1}/{MAX_RECONNECT_ATTEMPTS})...")
        cap.release()
        time.sleep(2)  # Wait before reconnecting
        cap = cv2.VideoCapture(STREAM_URL)
        reconnect_attempts += 1
        
        if reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
            print("❌ Failed to reconnect after multiple attempts.")
            break
        continue
    
    # Reset reconnect attempts on successful read
    if reconnect_attempts > 0:
        print("✓ Stream reconnected successfully")
        reconnect_attempts = 0
    
    # Rate limiting - only process frames at TARGET_FPS
    now = time.time()
    if now - last_capture_time < frame_interval:
        continue
    
    last_capture_time = now
    
    # Generate timestamp and filename
    t = datetime.datetime.now()
    filename = f"{t.year}-{t.month}-{t.day}-{t.hour}-{t.minute}-{t.second}.png"
    filepath = os.path.join("captures", filename)
    
    # Save raw frame (optimized)
    if SAVE_IMAGES:
        if USE_JPEG:
            filepath_jpg = filepath.replace(".png", ".jpg")
            cv2.imwrite(filepath_jpg, frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            print(f"Saved frame: {filepath_jpg}")
        else:
            cv2.imwrite(filepath, frame)
            print(f"Saved frame: {filepath}")
    else:
        print(f"Frame {frames_processed + 1} (saving disabled)")
    
    # Frame quality diagnostics
    if DEBUG_FRAMES:
        print(f"  Frame quality: shape={frame.shape}, dtype={frame.dtype}, "
              f"range=[{frame.min()}, {frame.max()}]")
        if frame.min() == frame.max():
            print("  ⚠ WARNING: Frame appears corrupted (all same value)")
        if frame.shape[0] < 480 or frame.shape[1] < 640:
            print(f"  ⚠ WARNING: Frame resolution very low ({frame.shape[1]}x{frame.shape[0]})")
    
    # YOLO tracking with optimized parameters for highway cams
    if DEBUG_TIMING:
        t0 = time.time()
    
    if ENABLE_TRACKING:
        results = model.track(
            source=frame,
            persist=True,  # Keep tracks across frames
            imgsz=INFERENCE_SIZE,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            classes=TRAFFIC_CLASSES if FILTER_CLASSES else None,
            tracker="bytetrack.yaml",  # ByteTrack for speed
            verbose=False
        )
    else:
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
        print(f"  Inference time: {inference_ms:.1f}ms ({1000/inference_ms:.1f} FPS max)")
    
    # Save annotated frame with enhanced visualization
    try:
        annotated_frame = result.plot(
            line_width=3,
            font_size=1.5,
            labels=True,
            conf=True,
            boxes=True
        )
    except TypeError:
        # Fallback if plot() parameters not supported in this version
        annotated_frame = result.plot()
        print("  (using default plot parameters)")
    
    annotated_path = os.path.join("captures_annotated", filename)
    if SAVE_IMAGES:
        if USE_JPEG:
            annotated_path_jpg = annotated_path.replace(".png", ".jpg")
            cv2.imwrite(annotated_path_jpg, annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            print(f"Saved annotated image: {annotated_path_jpg}")
        else:
            cv2.imwrite(annotated_path, annotated_frame)
            print(f"Saved annotated image: {annotated_path}")
    
    # Print detections with summary and track IDs
    boxes = result.boxes
    total_detections = len(boxes)
    high_conf_count = sum(1 for box in boxes if float(box.conf[0]) >= 0.5)
    
    # Get track IDs if tracking enabled
    track_ids = boxes.id if (ENABLE_TRACKING and boxes.id is not None) else None
    
    print(f"\nDetections in frame {frames_processed + 1}: {total_detections} total, {high_conf_count} high-confidence")
    
    if ENABLE_TRACKING and track_ids is not None:
        unique_tracks = len(set([int(tid) for tid in track_ids]))
        print(f"  Tracked objects: {unique_tracks} unique IDs")
    
    if not FILTER_CLASSES and total_detections == 0:
        print("  ℹ No detections even with all classes enabled - check frame quality or inference size")
    elif FILTER_CLASSES and total_detections == 0:
        print("  ℹ No traffic objects detected. Try setting FILTER_CLASSES=False to debug")
    
    if total_detections > 0:
        for i, box in enumerate(boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_name = result.names[cls]
            
            # Get track ID if available
            track_id = int(track_ids[i]) if track_ids is not None else None
            
            # Mark high-confidence detections
            marker = "★" if conf >= 0.5 else " "
            
            if track_id is not None:
                print(f"  {marker} ID:{track_id:3d} {class_name}: {conf:.2%} at ({x1:.0f}, {y1:.0f})")
            else:
                print(f"  {marker} {class_name}: {conf:.2%} at ({x1:.0f}, {y1:.0f})")
    print()
    
    frames_processed += 1

# -------- CLEANUP --------
cap.release()

# Calculate actual FPS achieved
if frames_processed > 0:
    print(f"\n✓ Processing complete!")
    print(f"✓ Processed {frames_processed} frames")
    if SAVE_IMAGES:
        if USE_JPEG:
            print("✓ Raw frames in: captures/ (JPEG)")
            print("✓ Annotated images in: captures_annotated/ (JPEG)")
        else:
            print("✓ Raw frames in: captures/")
            print("✓ Annotated images in: captures_annotated/")
    else:
        print("✓ Image saving disabled (benchmark mode)")
