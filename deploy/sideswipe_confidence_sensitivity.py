"""Post-hoc fixed-cutoff sensitivity on one synthetic scene, not calibration."""

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
from deploy.sideswipe_candidate_diagnostic import author_box, bbox_iou

TUNING_FRAMES = (126, 127, 140)
SAMPLE_FRAMES = tuple(i for i in range(0, 232, 10) if i not in TUNING_FRAMES)


def main():
    import cv2
    import numpy as np
    import torch
    import pipeline
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    from ultralytics.utils import nms, ops

    verify_source(verify_only=True)
    annotation = json.loads(Path("/media/annotation.json").read_text())
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    weights = Path("/models/yolo11n.pt")
    weights_sha = hashlib.sha256(weights.read_bytes()).hexdigest()
    if weights_sha != "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1":
        raise ValueError("Frozen weights changed")
    model = YOLO(str(weights)).to("cpu")
    model.add_callback("on_predict_start", lambda _: torch.set_num_threads(2))
    original_nms = nms.non_max_suppression
    original_postprocess = DetectionPredictor.postprocess
    observation = {}
    records = []

    def observe_nms(prediction, *args, **kwargs):
        raw = prediction[0] if isinstance(prediction, (tuple, list)) else prediction
        if raw.shape[0:2] != (1, 84) or args[:4] != (0.3, 0.45, [2, 3, 5, 7], False):
            raise ValueError("Production model/postprocessing contract changed")
        copy = raw.detach().clone()
        actual = original_nms(prediction, *args, **kwargs)
        replay = original_nms(copy.clone(), *args, **kwargs)
        if len(actual) != 1 or not torch.equal(actual[0], replay[0]):
            raise ValueError("Same-output production replay mismatch")
        exploratory = original_nms(copy.clone(), 0.1, *args[1:], **kwargs)
        if any(len(output[0]) >= kwargs["max_det"] for output in (actual, exploratory)):
            raise ValueError("Detection truncation prevents comparison")
        observation["outputs"] = {0.3: actual[0].detach().clone(),
                                  0.1: exploratory[0].detach().clone()}
        return actual

    def observe_postprocess(predictor, preds, img, orig_imgs, **kwargs):
        observation["tensor_shape"] = list(img.shape)
        observation["settings"] = {key: getattr(predictor.args, key) for key in
                                   ("conf", "iou", "classes", "agnostic_nms", "max_det", "imgsz", "half", "rect")}
        return original_postprocess(predictor, preds, img, orig_imgs, **kwargs)

    selected = set(TUNING_FRAMES + SAMPLE_FRAMES)
    started = time.monotonic()
    cap = cv2.VideoCapture("/media/video.mp4")
    try:
        with patch.object(nms, "non_max_suppression", observe_nms), patch.object(
            DetectionPredictor, "postprocess", observe_postprocess
        ):
            for index in range(232):
                ok, frame = cap.read()
                if not ok:
                    raise ValueError("Incomplete sequential decode")
                if index not in selected:
                    continue
                observation.clear()
                processed = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
                final = pipeline._run_yolo(model, processed)
                paths = []
                for confidence, output in observation["outputs"].items():
                    boxes = ops.scale_boxes(observation["tensor_shape"][2:], output[:, :4].clone(), processed.shape)
                    detections = [
                        {"xyxy_processed": box.tolist(),
                         "xyxy_source": [float(x) * 1.5 for x in box],
                         "confidence": float(output[i, 4]),
                         "class_id": int(output[i, 5]), "class_name": model.names[int(output[i, 5])]}
                        for i, box in enumerate(boxes)
                    ]
                    if confidence == 0.3:
                        if len(final) != len(detections) or any(
                            list(d[:4]) != b["xyxy_processed"] or d[4] != b["confidence"] or d[5] != b["class_name"]
                            for d, b in zip(final, detections)
                        ):
                            raise ValueError("Production final output mismatch")
                    targets = []
                    objects = annotation["base"][index]["objects"]
                    for actor in (448, 450):
                        present = any(obj["id"] == actor for obj in objects)
                        if not present:
                            targets.append({"actor_id": actor, "annotation_present": False,
                                            "best_iou": None, "matched_iou_0_50": None})
                            continue
                        target = author_box(objects, actor)
                        ious = [bbox_iou(box["xyxy_source"], target) for box in detections]
                        targets.append({"actor_id": actor, "annotation_present": True,
                                        "best_iou": max(ious, default=0),
                                        "matched_iou_0_50": any(iou >= 0.5 for iou in ious),
                                        "matching_detection_indices": [i for i, iou in enumerate(ious) if iou >= 0.5]})
                    paths.append({"confidence_cutoff": confidence, "vehicle_detection_count": len(detections),
                                  "detections": detections, "targets": targets})
                records.append({"source_frame": index, "decoded_pts_seconds": cap.get(cv2.CAP_PROP_POS_MSEC) / 1000,
                                "subset": "tuning_frames" if index in TUNING_FRAMES else "predetermined_sparse_sample",
                                "tensor_shape": observation["tensor_shape"],
                                "settings_except_exploratory_confidence": observation["settings"], "paths": paths})
    finally:
        cap.release()
    if {record["source_frame"] for record in records} != selected:
        raise ValueError("Sparse sample incomplete")
    aggregates = []
    for subset in ("tuning_frames", "predetermined_sparse_sample"):
        for cutoff in (0.3, 0.1):
            paths = [next(path for path in record["paths"] if path["confidence_cutoff"] == cutoff)
                     for record in records if record["subset"] == subset]
            aggregates.append({
                "subset": subset, "confidence_cutoff": cutoff, "frames": len(paths),
                "total_vehicle_detections": sum(path["vehicle_detection_count"] for path in paths),
                "actors": [{"actor_id": actor,
                            "annotated_observations": sum(t["annotation_present"] for p in paths
                                                         for t in p["targets"] if t["actor_id"] == actor),
                            "matched_observations": sum(t["matched_iou_0_50"] is True for p in paths
                                                       for t in p["targets"] if t["actor_id"] == actor)}
                           for actor in (448, 450)],
            })
    report = {
        "code_sha": subprocess.check_output(["git", "-C", "/source", "rev-parse", "HEAD"], text=True).strip(),
        "baseline": "b76b56a4444d4c35081e1cee5413341a019a59b7", "model_sha256": weights_sha,
        "packages": {key: importlib.metadata.version(key) for key in
                     ("torch", "torchvision", "ultralytics", "opencv-python", "numpy")},
        "python": platform.python_version(), "elapsed_seconds": time.monotonic() - started,
        "source_dimensions": [1920, 1080], "processed_dimensions": [1280, 720],
        "tuning_frames": TUNING_FRAMES, "predetermined_sparse_sample": SAMPLE_FRAMES,
        "observation_contract": "Same raw forward output for .30/.10; actual .30 output passed unchanged and replay checked equal; .10 is diagnostic postprocessing only. No labels to detector.",
        "limitations": "Post-hoc .10 cutoff selected after inspecting three tuning frames. Remaining sparse frames are from same synthetic scene, not independent generalization. IoU matches are geometric, not actor identity. Counts are not precision or false positives. No tracker/scorer inference or production changes.",
        "aggregates": aggregates, "records": records,
    }
    Path("/results/confidence-sensitivity.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
