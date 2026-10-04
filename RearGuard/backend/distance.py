# 23L-0549, 23L-0736, 23L-0848
# distance.py — Distance estimation using hybrid approach (area-ratio + focal length)

import numpy as np
from collections import deque

# Real-world heights (metres) per COCO class — used by the focal-length distance formula
REAL_HEIGHTS = {
    "car":           1.5,
    "truck":         2.5,
    "bus":           3.0,
    "motorbike":     1.1,
    "bicycle":       1.1,
    "person":        1.7,
    "dog":           0.5,
    "cat":           0.3,
    "traffic light": 0.9,
    "stop sign":     0.9,
    "laptop":        0.3,
    "chair":         0.9,
    "bottle":        0.3,
    "suitcase":      0.7,
    "backpack":      0.5,
    "umbrella":      1.0,
    "default":       1.2,
}

DEFAULT_FOCAL_LENGTH_PX = 800  # Default focal length in pixels; calibrate for your camera


def estimate_distance_area(box_w, box_h, frame_w, frame_h):
    # area ratio based distance approximation using a lookup table
    frame_area = frame_w * frame_h
    if frame_area <= 0:
        return 20.0

    ratio = (box_w * box_h) / frame_area

    if ratio > 0.55:   return 0.4
    elif ratio > 0.40: return 0.8
    elif ratio > 0.30: return 1.2
    elif ratio > 0.20: return 1.8
    elif ratio > 0.12: return 2.8
    elif ratio > 0.07: return 4.0
    elif ratio > 0.03: return 6.5
    elif ratio > 0.01: return 10.0
    else:              return 16.0


def estimate_distance_focal(pixel_height, label="default",
                             focal_length_px=DEFAULT_FOCAL_LENGTH_PX):
    # pinhole camera formula: D = (H_real * f) / H_pixels
    if pixel_height <= 0:
        return 20.0
    real_h = REAL_HEIGHTS.get(label, REAL_HEIGHTS["default"])
    return min(round((real_h * focal_length_px) / pixel_height, 2), 20.0)


def estimate_distance(pixel_height, box_w, box_h, frame_w, frame_h,
                      label="default",
                      focal_length_px=DEFAULT_FOCAL_LENGTH_PX):
    # combine both methods 70% area ratio + 30% focal length
    d_area  = estimate_distance_area(box_w, box_h, frame_w, frame_h)
    d_focal = estimate_distance_focal(pixel_height, label, focal_length_px)
    return round(min((d_area * 0.70) + (d_focal * 0.30), 20.0), 2)


def get_zone(distance, danger_m=3.0, warning_m=8.0):
    # returns danger/warning/safe based on distance thresholds
    if distance >= 18.0: return "no object"
    if distance <= danger_m: return "danger"
    if distance <= warning_m: return "warning"
    return "safe"


class SmoothingFilter:
    # median filter to smooth out jumpy distance values
    def __init__(self, window=5):
        self._buf = deque(maxlen=window)

    def update(self, value):
        self._buf.append(value)
        return float(np.median(self._buf))

    def reset(self):
        self._buf.clear()