"""
Central configuration for the DeepSORT tracking + accident confidence pipeline.

All tunable parameters live here. Import what you need from other modules:
    from config import DEEPSORT_MAX_AGE, VEHICLE_CLASSES, ...
"""

# ---------------------------------------------------------------------------
# DeepSORT tracker parameters
# ---------------------------------------------------------------------------
DEEPSORT_MAX_AGE          = 30    # Frames a track survives without a matching detection
DEEPSORT_N_INIT           = 3     # Detections needed before a track is confirmed
DEEPSORT_MAX_IOU_DISTANCE = 0.7   # IoU gating distance for assignment
DEEPSORT_MAX_COSINE_DIST  = 0.3   # Re-ID embedding cosine distance threshold
DEEPSORT_NN_BUDGET        = 100   # Max appearance descriptors kept per track (None = unlimited)

# ---------------------------------------------------------------------------
# Detection / YOLO parameters
# ---------------------------------------------------------------------------
# COCO class IDs for vehicles only (car=2, motorcycle=3, bus=5, truck=7)
VEHICLE_CLASSES = [2, 3, 5, 7]

# Minimum YOLO confidence to pass a detection to the tracker
DET_CONF_THRESHOLD = 0.30

# YOLO model to use (auto-downloaded on first run)
YOLO_MODEL = "yolo11x.pt"

# Inference image size (larger = better small-object detection, slower)
INFERENCE_SIZE = 1280

# NMS IoU threshold for YOLO post-processing
IOU_THRESHOLD = 0.45

# ---------------------------------------------------------------------------
# History / rolling time window
# ---------------------------------------------------------------------------
# Assumed frame rate used to convert seconds → frames when the real FPS is
# not yet known (e.g. first few frames of a stream).  Pipeline overrides this
# with the measured stream FPS once it is available.
ASSUMED_FPS = 10

# How many seconds of track history to keep per vehicle
HISTORY_WINDOW_SECONDS = 5

# Derived: history window in frames (recalculated in pipeline with real FPS)
HISTORY_WINDOW_FRAMES = ASSUMED_FPS * HISTORY_WINDOW_SECONDS  # 50 frames default

# How many frames to keep a "lost" track's history after it disappears
# (grace period allows scoring even if detection blinks out for a moment)
GRACE_PERIOD_FRAMES = 10

# ---------------------------------------------------------------------------
# Signal: sudden stop
# ---------------------------------------------------------------------------
# Speed (pixels/frame) below which a track is considered stopped
STOP_SPEED_THRESHOLD = 5.0

# How many consecutive frames the speed must stay below threshold
STOP_MIN_FRAMES = 5

# ---------------------------------------------------------------------------
# Signal: abrupt deceleration
# ---------------------------------------------------------------------------
# Mean acceleration (pixels/frame²) below which deceleration is flagged
# (negative = deceleration)
DECEL_THRESHOLD = -3.0

# Number of recent frames over which mean acceleration is computed
DECEL_WINDOW_FRAMES = 8

# ---------------------------------------------------------------------------
# Signal: collision / box overlap
# ---------------------------------------------------------------------------
# Minimum IoU between two tracks' boxes to trigger a collision event
COLLISION_IOU_THRESHOLD = 0.10

# Minimum number of frames over which centre-distance must shrink
# before an IoU overlap is counted as an approach-then-collide event
COLLISION_APPROACH_FRAMES = 6

# ---------------------------------------------------------------------------
# Signal: post-collision behaviour
# ---------------------------------------------------------------------------
# Speed threshold below which a track is "stationary" after a collision
POST_COLLISION_STATIONARY_SPEED = 4.0

# Direction change (degrees) considered "erratic" post-collision
POST_COLLISION_DIRECTION_CHANGE_DEG = 70.0

# How many frames after an overlap to look for post-collision behaviour
POST_COLLISION_WINDOW_FRAMES = 15

# ---------------------------------------------------------------------------
# Signal: traffic anomaly (nearby vehicles also slowing)
# ---------------------------------------------------------------------------
# Pixel radius around the event region to search for affected tracks
ANOMALY_RADIUS_PX = 250

# Minimum number of additional stopped/slow tracks to trigger the signal
ANOMALY_MIN_SLOW_TRACKS = 2

# ---------------------------------------------------------------------------
# Scoring weights  (must sum to 1.0)
# ---------------------------------------------------------------------------
WEIGHT_SUDDEN_STOP     = 0.25
WEIGHT_DECEL           = 0.20
WEIGHT_COLLISION       = 0.30
WEIGHT_POST_COLLISION  = 0.15
WEIGHT_TRAFFIC_ANOMALY = 0.10

# Quick sanity-check (helps catch edit mistakes)
_WEIGHT_SUM = (
    WEIGHT_SUDDEN_STOP + WEIGHT_DECEL + WEIGHT_COLLISION
    + WEIGHT_POST_COLLISION + WEIGHT_TRAFFIC_ANOMALY
)
assert abs(_WEIGHT_SUM - 1.0) < 1e-6, (
    f"Scoring weights must sum to 1.0, got {_WEIGHT_SUM:.4f}"
)

# ---------------------------------------------------------------------------
# EMA smoothing
# ---------------------------------------------------------------------------
# Exponential moving average alpha for the accident score.
# Higher = faster response, lower = smoother/less noisy.
EMA_ALPHA = 0.30

# ---------------------------------------------------------------------------
# Hysteresis thresholds for accident_detected flag
# ---------------------------------------------------------------------------
# Score must rise above HIGH_THRESHOLD to set accident_detected = True
HIGH_THRESHOLD = 0.65

# Score must fall below LOW_THRESHOLD to clear accident_detected back to False
LOW_THRESHOLD = 0.40

# ---------------------------------------------------------------------------
# Output / saving
# ---------------------------------------------------------------------------
# Directory for annotated output frames
OUTPUT_DIR = "pipeline_output"

# Save annotated frames to disk (set False to benchmark speed)
SAVE_FRAMES = True

# Use JPEG (faster) vs PNG for saved frames
USE_JPEG = True
JPEG_QUALITY = 85

# Print per-frame score to console
VERBOSE = True

# Overlay the accident score on annotated frames
OVERLAY_SCORE = True
