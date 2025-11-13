"""
Example script showing how to access and analyze YOLO detection data
for future accident detection logic.

This demonstrates:
1. Extracting detection data (classes, confidence, positions)
2. Filtering for specific vehicle types
3. Basic analysis that could lead to accident detection
"""

from ultralytics import YOLO
import cv2
import os
import glob

# Load model
model = YOLO("yolov8n.pt")

# Get all images from captures_annotated folder
captures_folder = "captures_annotated"
image_files = glob.glob(os.path.join(captures_folder, "*.png"))

if not image_files:
    print(f"Error: No images found in {captures_folder}/ folder")
    print("Please run yolo_screenshot_detector.py first to capture some images")
    exit()

print(f"Found {len(image_files)} images to analyze\n")
print("=" * 80)

# Store aggregate statistics
total_vehicles_all_frames = 0
total_overlaps_all_frames = 0
all_frame_stats = []

# Process each image
for img_idx, image_path in enumerate(sorted(image_files), 1):
    print(f"\n{'=' * 80}")
    print(f"ANALYZING IMAGE {img_idx}/{len(image_files)}: {os.path.basename(image_path)}")
    print("=" * 80)
    
    try:
        results = model(image_path)
        result = results[0]

        # -------- ACCESSING DETECTION DATA --------
        boxes = result.boxes
        
        print(f"\nTotal detections: {len(boxes)}")
        
        # YOLOv8 COCO dataset class IDs (relevant for traffic)
        VEHICLE_CLASSES = {
            2: "car",
            3: "motorcycle", 
            5: "bus",
            7: "truck",
            0: "person"
        }
        
        vehicle_detections = []
        
        for idx, box in enumerate(boxes):
            cls = int(box.cls[0])        # class id
            conf = float(box.conf[0])    # confidence
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # box coordinates
            
            # Get class name
            class_name = result.names[cls]
            
            print(f"  Detection {idx + 1}: {class_name} (conf: {conf:.2f})")
            
            # Store vehicle detections for analysis
            if cls in VEHICLE_CLASSES:
                vehicle_detections.append({
                    'class': class_name,
                    'class_id': cls,
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2),
                    'center': ((x1+x2)/2, (y1+y2)/2),
                    'area': (x2-x1) * (y2-y1)
                })
        
        # -------- EXAMPLE ANALYSIS FOR ACCIDENT DETECTION --------
        print(f"\n--- Analysis ---")
        print(f"Vehicle count: {len(vehicle_detections)}")
        
        # Example 1: Check for overlapping vehicles (potential collision)
        def boxes_overlap(box1, box2, threshold=0.2):
            """Check if two bounding boxes overlap significantly"""
            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2
            
            # Calculate intersection
            x_left = max(x1_1, x1_2)
            y_top = max(y1_1, y1_2)
            x_right = min(x2_1, x2_2)
            y_bottom = min(y2_1, y2_2)
            
            if x_right < x_left or y_bottom < y_top:
                return False, 0
            
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
            box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
            
            # IoU (Intersection over Union)
            union_area = box1_area + box2_area - intersection_area
            iou = intersection_area / union_area if union_area > 0 else 0
            
            return iou > threshold, iou
        
        overlap_count = 0
        overlaps_found = False
        for i in range(len(vehicle_detections)):
            for j in range(i + 1, len(vehicle_detections)):
                v1 = vehicle_detections[i]
                v2 = vehicle_detections[j]
                overlaps, iou = boxes_overlap(v1['bbox'], v2['bbox'])
                if overlaps:
                    print(f"  ⚠️  OVERLAP: {v1['class']} and {v2['class']} (IoU: {iou:.2f})")
                    overlaps_found = True
                    overlap_count += 1
        
        if not overlaps_found:
            print("  ✓ No significant overlaps detected")
        
        # High-confidence vehicle count
        high_conf_vehicles = [v for v in vehicle_detections if v['confidence'] > 0.7]
        print(f"  High-confidence vehicles (>0.7): {len(high_conf_vehicles)}")
        
        # Store frame statistics
        frame_stats = {
            'filename': os.path.basename(image_path),
            'total_detections': len(boxes),
            'vehicle_count': len(vehicle_detections),
            'overlaps': overlap_count,
            'high_conf_count': len(high_conf_vehicles)
        }
        all_frame_stats.append(frame_stats)
        total_vehicles_all_frames += len(vehicle_detections)
        total_overlaps_all_frames += overlap_count
    
    except FileNotFoundError:
        print(f"  ❌ Error: Image not found at {image_path}")
    except Exception as e:
        print(f"  ❌ Error processing image: {e}")

# -------- AGGREGATE SUMMARY --------
print("\n" + "=" * 80)
print("SUMMARY ACROSS ALL FRAMES")
print("=" * 80)

print(f"\nTotal images analyzed: {len(all_frame_stats)}")
print(f"Total vehicles detected: {total_vehicles_all_frames}")
print(f"Total overlaps detected: {total_overlaps_all_frames}")

if all_frame_stats:
    avg_vehicles = total_vehicles_all_frames / len(all_frame_stats)
    print(f"Average vehicles per frame: {avg_vehicles:.1f}")
    
    # Find frame with most vehicles
    max_frame = max(all_frame_stats, key=lambda x: x['vehicle_count'])
    print(f"\nFrame with most vehicles: {max_frame['filename']} ({max_frame['vehicle_count']} vehicles)")
    
    # Find frames with overlaps (potential accidents)
    frames_with_overlaps = [f for f in all_frame_stats if f['overlaps'] > 0]
    if frames_with_overlaps:
        print(f"\n⚠️  Frames with overlaps (potential issues): {len(frames_with_overlaps)}")
        for frame in frames_with_overlaps:
            print(f"  - {frame['filename']}: {frame['overlaps']} overlap(s)")
    else:
        print(f"\n✓ No overlaps detected in any frame")

print("\n" + "=" * 80)
print("NOTES FOR FUTURE ACCIDENT DETECTION:")
print("=" * 80)
print("""
1. Frame-to-frame tracking:
   - Track vehicle positions across multiple frames
   - Detect sudden stops or unusual movements
   - Identify vehicles that remain stationary in unexpected locations

2. Anomaly detection:
   - Compare current frame to baseline "normal traffic"
   - Detect unusual clustering of vehicles
   - Identify debris or people on roadway

3. Custom model training:
   - Train YOLOv8 on accident-specific dataset
   - Binary classification: "accident" vs "normal"
   - Or multi-class: types of incidents

4. Multi-frame analysis:
   - Store last N frames
   - Analyze motion patterns
   - Detect rapid changes in vehicle count or positions
""")

