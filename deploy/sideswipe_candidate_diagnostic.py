"""Observe three fixed YOLO outputs without changing prediction/postprocessing."""

import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, "/source")

from deploy.carla_experiment import main as verify_source


def bbox_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    if union <= 0:
        raise ValueError("Invalid diagnostic box area")
    return intersection / union


def author_box(objects, actor):
    matches = [obj for obj in objects if obj["id"] == actor]
    if len(matches) != 1:
        raise ValueError(f"Expected unique actor {actor}")
    box = matches[0]["2d_bbox"]
    if len(box) == 2 and all(len(point) == 2 for point in box):
        box = [*box[0], *box[1]]
    if len(box) != 4 or not all(isinstance(x, (int, float)) for x in box):
        raise ValueError("Unsupported author bbox schema; do not guess")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Invalid author bbox")
    return box


def candidate_status(confidence, class_id, kept, vehicle_classes, threshold):
    if confidence <= threshold:
        return "confidence_filtered"
    if class_id not in vehicle_classes:
        return "class_filtered"
    return "retained" if kept else "nms_removed"


def main():
    import cv2
    import numpy as np
    import torch
    import config
    import pipeline
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    from ultralytics.utils import nms, ops

    verify_source(verify_only=True)
    annotation = json.loads(Path("/media/annotation.json").read_text())
    frames = {}
    cap = cv2.VideoCapture("/media/video.mp4")
    try:
        for index in range(232):
            ok, frame = cap.read()
            if not ok:
                raise ValueError("Sequential diagnostic decode incomplete")
            if index in (126, 127, 140):
                frames[index] = frame
    finally:
        cap.release()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model = YOLO("/models/yolo11n.pt").to("cpu")
    model.add_callback("on_predict_start", lambda _: torch.set_num_threads(2))
    if hashlib.sha256(Path("/models/yolo11n.pt").read_bytes()).hexdigest() != (
        "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
    ):
        raise ValueError("Frozen weights changed")
    original_postprocess = DetectionPredictor.postprocess
    original_nms = nms.non_max_suppression
    records = []
    observation = {}

    def observe_nms(prediction, *args, **kwargs):
        raw = prediction[0] if isinstance(prediction, (tuple, list)) else prediction
        if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[1] != 84:
            raise ValueError("Unsupported raw prediction shape")
        untouched = raw.detach().clone()
        output = original_nms(prediction, *args, **kwargs)
        production_output = [item.detach().clone() for item in output]
        replay, indices = original_nms(untouched.clone(), *args, **{**kwargs, "return_idxs": True})
        if len(output) != 1 or not torch.equal(production_output[0], replay[0]):
            raise ValueError("Diagnostic replay differs from actual NMS")
        if len(replay[0]) >= kwargs["max_det"] or raw.shape[-1] > 30000:
            raise ValueError("NMS candidate/detection truncation prevents attribution")
        observation.update(raw=untouched, kept=set(indices[0].tolist()),
                           output=production_output[0])
        return output

    def observe_postprocess(predictor, preds, img, orig_imgs, **kwargs):
        observation["tensor_shape"] = list(img.shape)
        observation["settings"] = {key: getattr(predictor.args, key) for key in
                                   ("conf", "iou", "classes", "agnostic_nms", "max_det", "imgsz", "half", "rect")}
        return original_postprocess(predictor, preds, img, orig_imgs, **kwargs)

    started = time.monotonic()
    with patch.object(nms, "non_max_suppression", observe_nms), patch.object(
        DetectionPredictor, "postprocess", observe_postprocess
    ):
        for index, frame in frames.items():
            observation.clear()
            processed = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
            final = pipeline._run_yolo(model, processed)
            raw = observation["raw"][0].transpose(0, 1)
            boxes = ops.xywh2xyxy(raw[:, :4])
            boxes = ops.scale_boxes(observation["tensor_shape"][2:], boxes, processed.shape) * 1.5
            confidences, classes = raw[:, 4:].max(1)
            actors = []
            for actor in (448, 450):
                target = author_box(annotation["base"][index]["objects"], actor)
                candidates = []
                for candidate, box in enumerate(boxes.tolist()):
                    # Degenerate low-confidence predictions cannot match a valid target.
                    if box[2] <= box[0] or box[3] <= box[1]:
                        continue
                    confidence, class_id = float(confidences[candidate]), int(classes[candidate])
                    candidates.append({
                        "candidate_index": candidate, "xyxy_source": box,
                        "iou_to_actor": bbox_iou(box, target), "confidence": confidence,
                        "class_id": class_id, "class_name": model.names[class_id],
                        "car_confidence": float(raw[candidate, 6]),
                        "status": candidate_status(confidence, class_id,
                                                   candidate in observation["kept"],
                                                   config.VEHICLE_CLASSES, config.DET_CONF_THRESHOLD),
                    })
                matching = sorted((c for c in candidates if c["iou_to_actor"] >= 0.5),
                                  key=lambda c: c["confidence"], reverse=True)
                for candidate in matching:
                    if candidate["status"] == "nms_removed":
                        suppressors = [c for c in candidates if c["status"] == "retained"
                                       and c["class_id"] == candidate["class_id"]
                                       and c["confidence"] >= candidate["confidence"]
                                       and bbox_iou(c["xyxy_source"], candidate["xyxy_source"]) > config.IOU_THRESHOLD]
                        candidate["possible_retained_suppressors"] = [
                            {"candidate_index": c["candidate_index"],
                             "confidence": c["confidence"],
                             "pair_iou": bbox_iou(c["xyxy_source"], candidate["xyxy_source"])}
                            for c in suppressors]
                        if not suppressors:
                            raise ValueError("NMS removal lacks a retained suppressor; attribution ambiguous")
                eligible = [c for c in candidates if c["status"] in ("retained", "nms_removed")]
                final_ious = [bbox_iou([x * 1.5 for x in d[:4]], target) for d in final]
                actors.append({
                    "actor_id": actor, "raw_candidate_count": len(candidates),
                    "matching_candidate_count": len(matching),
                    "matching_status_counts": {status: sum(c["status"] == status for c in matching)
                                               for status in ("confidence_filtered", "class_filtered", "nms_removed", "retained")},
                    "best_raw_iou": max(candidates, key=lambda c: c["iou_to_actor"]),
                    "best_eligible_iou": max(eligible, key=lambda c: c["iou_to_actor"]) if eligible else None,
                    "best_final_vehicle_iou": max(final_ious, default=0),
                    "matching_candidates_top20_by_confidence": matching[:20],
                    "candidate_export_limit": 20,
                })
            records.append({
                "source_frame": index, "pts_seconds": index / 20,
                "tensor_shape": observation["tensor_shape"], "settings": observation["settings"],
                "raw_prediction_shape": list(observation["raw"].shape),
                "final_detections": [{"xyxy_processed": list(d[:4]), "confidence": d[4],
                                      "class_name": d[5]} for d in final],
                "actors": actors,
            })
    report = {
        "code_sha": subprocess.check_output(["git", "-C", "/source", "rev-parse", "HEAD"], text=True).strip(),
        "baseline": "b76b56a4444d4c35081e1cee5413341a019a59b7",
        "packages": {key: importlib.metadata.version(key) for key in
                     ("torch", "torchvision", "ultralytics", "opencv-python", "numpy")},
        "python": platform.python_version(), "elapsed_seconds": time.monotonic() - started,
        "source_dimensions": [1920, 1080], "processed_dimensions": [1280, 720],
        "coordinate_system": "xyxy source pixels; processed source scale 1.5; INTER_LINEAR then normal YOLO letterbox",
        "observation_contract": "Original predictions/NMS outputs returned unchanged; exact index-return replay checked equal. No labels fed to inference. Three stateless detector calls, no tracker/scorer rerun.",
        "limitations": "IoU attribution is geometric, not actor identity or general accuracy. Possible suppressors are retained same-class higher-confidence overlapping boxes, not an instrumented internal NMS causal trace. No threshold tuning.",
        "records": records,
    }
    Path("/results/candidate-diagnostic.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
