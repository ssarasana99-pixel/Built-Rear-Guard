# 23L-0549, 23L-0736, 23L-0848
from ultralytics import YOLO
import numpy as np

_model = None

# Classes to skip
SKIP_CLASSES = {
    "toothbrush", "fork", "knife", "spoon", "scissors",
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake",
    "wine glass", "cup", "bowl",
}

ZONE_COLORS = {
    "danger":    (0,   0,   255),
    "warning":   (0,   165, 255),
    "safe":      (0,   255, 0),
    "no object": (200, 200, 200),
}


def _load_model():
    global _model
    _model = YOLO("yolov8s.pt")
    print("[YOLOv8] Model loaded successfully")


def detect_obstacles(frame, conf_thresh=0.25):
    global _model
    if _model is None:
        _load_model()

    frame_h, frame_w = frame.shape[:2]

    results = _model(frame, conf=conf_thresh, verbose=False)[0]

    detections = []
    for box in results.boxes:
        class_id   = int(box.cls[0])
        confidence = float(box.conf[0])
        label      = _model.names[class_id]

        if label in SKIP_CLASSES:
            continue

        # Convert YOLO center-format (cx, cy, w, h) → top-left corner (x, y, w, h)
        cx, cy, w, h = box.xywh[0].tolist()
        x = int(cx - w / 2)
        y = int(cy - h / 2)
        w, h = int(w), int(h)

        # Skip tiny detections likely false positives occupying < 0.5% of frame
        if (w * h) < (frame_w * frame_h * 0.005):
            continue

        detections.append((x, y, w, h, label, round(confidence, 2)))

    # Sort by area descending so the closest/largest object is index 0
    detections.sort(key=lambda d: d[2] * d[3], reverse=True)

    print(f"[detect] {len(detections)} object(s): "
          f"{[(d[4], d[5]) for d in detections]}")

    return detections   # (x, y, w, h, label, confidence)


def draw_detections(frame, obstacles, distances, zone):
    """Draw bounding boxes, labels, distances, and zone indicator on the frame."""
    import cv2
    output = frame.copy()
    color  = ZONE_COLORS.get(zone, (255, 255, 255))

    for idx, det in enumerate(obstacles):
        x, y, w, h, label = det[0], det[1], det[2], det[3], det[4]
        conf = det[5] if len(det) > 5 else 0.0
        dist = distances[idx] if idx < len(distances) else None

        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

        text = f"{label} {conf:.0%}"
        if dist is not None:
            text += f"  {dist:.1f}m"

        cv2.putText(output, text, (x, max(y - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    cv2.putText(output, f"ZONE: {zone.upper()}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                ZONE_COLORS.get(zone, (255, 255, 255)), 3)

    return output